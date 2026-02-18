"""Tests for swarm manifest generation and P2P transfer."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tightwad.manifest import (
    PieceBitfield,
    PieceInfo,
    SwarmManifest,
    create_manifest,
    verify_piece,
    DEFAULT_PIECE_SIZE,
)
from tightwad.swarm_transfer import SwarmPuller, PeerState


# --- Fixtures ---


@pytest.fixture
def sample_file(tmp_path):
    """Create a 256KB test file with known content."""
    f = tmp_path / "test-model.gguf"
    # 4 pieces at 64KB each
    data = b""
    for i in range(4):
        data += bytes([i & 0xFF]) * (64 * 1024)
    f.write_bytes(data)
    return f


@pytest.fixture
def small_piece_size():
    return 64 * 1024  # 64KB for tests


@pytest.fixture
def sample_manifest(sample_file, small_piece_size):
    return create_manifest(sample_file, piece_size=small_piece_size, use_gguf_inspect=False)


@pytest.fixture
def sample_pieces():
    return [
        PieceInfo(index=0, offset=0, size=100, sha256="aaa"),
        PieceInfo(index=1, offset=100, size=100, sha256="bbb"),
        PieceInfo(index=2, offset=200, size=100, sha256="ccc"),
        PieceInfo(index=3, offset=300, size=50, sha256="ddd"),
    ]


# --- Manifest Generation ---


class TestManifestCreation:
    def test_piece_count(self, sample_file, small_piece_size):
        m = create_manifest(sample_file, piece_size=small_piece_size, use_gguf_inspect=False)
        assert m.num_pieces == 4

    def test_piece_sizes(self, sample_manifest):
        for piece in sample_manifest.pieces:
            assert piece.size == 64 * 1024

    def test_piece_offsets(self, sample_manifest):
        expected_offsets = [i * 64 * 1024 for i in range(4)]
        actual_offsets = [p.offset for p in sample_manifest.pieces]
        assert actual_offsets == expected_offsets

    def test_piece_sha256(self, sample_file, sample_manifest):
        # Verify first piece hash manually
        with open(sample_file, "rb") as f:
            chunk = f.read(64 * 1024)
        expected = hashlib.sha256(chunk).hexdigest()
        assert sample_manifest.pieces[0].sha256 == expected

    def test_last_piece_smaller(self, tmp_path):
        """Last piece should be smaller when file isn't evenly divisible."""
        f = tmp_path / "odd.gguf"
        f.write_bytes(b"\x42" * 150)
        m = create_manifest(f, piece_size=100, use_gguf_inspect=False)
        assert m.num_pieces == 2
        assert m.pieces[0].size == 100
        assert m.pieces[1].size == 50

    def test_total_size(self, sample_file, sample_manifest):
        assert sample_manifest.total_size == sample_file.stat().st_size

    def test_progress_callback(self, sample_file, small_piece_size):
        calls = []

        def cb(done, total):
            calls.append((done, total))

        create_manifest(sample_file, piece_size=small_piece_size, use_gguf_inspect=False, progress_callback=cb)
        assert len(calls) == 4
        assert calls[0][0] == 1
        assert calls[3][0] == 4

    def test_model_name_from_filename(self, sample_file, sample_manifest):
        assert sample_manifest.model == "test-model"
        assert sample_manifest.filename == "test-model.gguf"


# --- Manifest Serialization ---


class TestManifestSerialization:
    def test_to_dict_from_dict_roundtrip(self, sample_manifest):
        d = sample_manifest.to_dict()
        restored = SwarmManifest.from_dict(d)
        assert restored.model == sample_manifest.model
        assert restored.filename == sample_manifest.filename
        assert restored.total_size == sample_manifest.total_size
        assert restored.piece_size == sample_manifest.piece_size
        assert restored.num_pieces == sample_manifest.num_pieces
        for orig, rest in zip(sample_manifest.pieces, restored.pieces):
            assert orig.index == rest.index
            assert orig.offset == rest.offset
            assert orig.size == rest.size
            assert orig.sha256 == rest.sha256

    def test_save_load_roundtrip(self, sample_manifest, tmp_path):
        path = tmp_path / "manifest.json"
        sample_manifest.save(path)
        loaded = SwarmManifest.load(path)
        assert loaded.model == sample_manifest.model
        assert loaded.num_pieces == sample_manifest.num_pieces
        assert loaded.pieces[0].sha256 == sample_manifest.pieces[0].sha256

    def test_find_for_model(self, sample_file, sample_manifest):
        manifest_path = sample_file.parent / f"{sample_file.name}.tightwad.manifest"
        sample_manifest.save(manifest_path)
        found = SwarmManifest.find_for_model(sample_file)
        assert found is not None
        assert found.model == sample_manifest.model

    def test_find_for_model_missing(self, tmp_path):
        result = SwarmManifest.find_for_model(tmp_path / "nonexistent.gguf")
        assert result is None


# --- Piece Verification ---


class TestPieceVerification:
    def test_valid_piece(self, sample_file, sample_manifest):
        assert verify_piece(sample_file, sample_manifest.pieces[0]) is True
        assert verify_piece(sample_file, sample_manifest.pieces[3]) is True

    def test_corrupted_piece(self, sample_file, sample_manifest):
        piece = sample_manifest.pieces[0]
        bad_piece = PieceInfo(
            index=piece.index,
            offset=piece.offset,
            size=piece.size,
            sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )
        assert verify_piece(sample_file, bad_piece) is False


# --- Bitfield ---


class TestBitfield:
    def test_empty_start(self, tmp_path):
        bf = PieceBitfield.load_or_create(tmp_path / "pieces.json", total_pieces=10)
        assert len(bf.have) == 0
        assert bf.completion_pct() == 0.0

    def test_mark_have(self, tmp_path):
        bf = PieceBitfield.load_or_create(tmp_path / "pieces.json", total_pieces=4)
        bf.mark_have(0)
        bf.mark_have(2)
        assert 0 in bf.have
        assert 2 in bf.have
        assert 1 not in bf.have

    def test_mark_missing(self, tmp_path):
        bf = PieceBitfield.load_or_create(tmp_path / "pieces.json", total_pieces=4)
        bf.mark_have(0)
        bf.mark_have(1)
        bf.mark_missing(0)
        assert 0 not in bf.have
        assert 1 in bf.have

    def test_save_load_persistence(self, tmp_path):
        path = tmp_path / "pieces.json"
        bf = PieceBitfield.load_or_create(path, total_pieces=4)
        bf.mark_have(1)
        bf.mark_have(3)
        bf.save()

        bf2 = PieceBitfield.load_or_create(path, total_pieces=4)
        assert bf2.have == {1, 3}

    def test_completion_pct(self, tmp_path):
        bf = PieceBitfield.load_or_create(tmp_path / "p.json", total_pieces=4)
        assert bf.completion_pct() == 0.0
        bf.mark_have(0)
        assert bf.completion_pct() == 25.0
        bf.mark_have(1)
        bf.mark_have(2)
        bf.mark_have(3)
        assert bf.completion_pct() == 100.0

    def test_have_all(self, tmp_path):
        bf = PieceBitfield.load_or_create(tmp_path / "p.json", total_pieces=3)
        assert bf.have_all() is False
        bf.mark_have(0)
        bf.mark_have(1)
        assert bf.have_all() is False
        bf.mark_have(2)
        assert bf.have_all() is True

    def test_missing_pieces(self, tmp_path):
        bf = PieceBitfield.load_or_create(tmp_path / "p.json", total_pieces=4)
        bf.mark_have(1)
        bf.mark_have(3)
        assert bf.missing_pieces() == [0, 2]


# --- Piece Selection ---


class TestPieceSelection:
    def test_rarest_first(self):
        """Pieces available from fewer peers should be selected first."""
        peers = [
            PeerState(url="http://a", have={0, 1, 2, 3}),
            PeerState(url="http://b", have={0, 2}),
        ]
        puller = SwarmPuller(
            model_path=Path("/fake"),
            manifest=MagicMock(num_pieces=4, pieces=[]),
            bitfield=MagicMock(),
            peers=[],
        )
        puller.peers = peers
        missing = [0, 1, 2, 3]
        plan = puller._select_piece_order(peers, missing)

        # Piece 1 and 3 are rarest (only peer A has them)
        piece_order = [idx for idx, _ in plan]
        # 1 and 3 should come before 0 and 2
        assert piece_order.index(1) < piece_order.index(0)
        assert piece_order.index(3) < piece_order.index(0)

    def test_load_balancing(self):
        """Pieces should be distributed across peers."""
        peers = [
            PeerState(url="http://a", have={0, 1}),
            PeerState(url="http://b", have={0, 1}),
        ]
        puller = SwarmPuller(
            model_path=Path("/fake"),
            manifest=MagicMock(num_pieces=2, pieces=[]),
            bitfield=MagicMock(),
            peers=[],
        )
        puller.peers = peers
        plan = puller._select_piece_order(peers, [0, 1])

        # Both pieces equally rare — they should be spread across peers
        assigned_peers = [peer.url for _, peer in plan]
        assert len(plan) == 2

    def test_empty_missing(self):
        peers = [PeerState(url="http://a", have={0})]
        puller = SwarmPuller(
            model_path=Path("/fake"),
            manifest=MagicMock(),
            bitfield=MagicMock(),
            peers=[],
        )
        plan = puller._select_piece_order(peers, [])
        assert plan == []

    def test_no_peers(self):
        puller = SwarmPuller(
            model_path=Path("/fake"),
            manifest=MagicMock(),
            bitfield=MagicMock(),
            peers=[],
        )
        plan = puller._select_piece_order([], [0, 1])
        assert plan == []


# --- Download Piece ---


class TestDownloadPiece:
    @pytest.mark.asyncio
    async def test_successful_download(self, tmp_path):
        """Successful download with SHA256 verification."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 200)

        data = b"\xAB" * 100
        sha = hashlib.sha256(data).hexdigest()
        piece = PieceInfo(index=0, offset=0, size=100, sha256=sha)

        bf = PieceBitfield.load_or_create(tmp_path / "pieces.json", total_pieces=1)
        peer = PeerState(url="http://fake-peer:9080", have={0})

        puller = SwarmPuller(
            model_path=model_file,
            manifest=MagicMock(total_size=200, num_pieces=1, pieces=[piece]),
            bitfield=bf,
            peers=[],
        )

        mock_response = MagicMock()
        mock_response.content = data
        mock_response.raise_for_status = MagicMock()

        with patch("tightwad.swarm_transfer.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            sem = __import__("asyncio").Semaphore(4)
            ok = await puller.download_piece(piece, peer, sem)

        assert ok is True
        assert 0 in bf.have

        # Verify data written at correct offset
        with open(model_file, "rb") as f:
            written = f.read(100)
        assert written == data

    @pytest.mark.asyncio
    async def test_corrupted_piece_rejected(self, tmp_path):
        """Piece with wrong SHA256 should be rejected."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 100)

        piece = PieceInfo(index=0, offset=0, size=100, sha256="expected_but_wrong")

        bf = PieceBitfield.load_or_create(tmp_path / "pieces.json", total_pieces=1)
        peer = PeerState(url="http://fake-peer:9080", have={0})

        puller = SwarmPuller(
            model_path=model_file,
            manifest=MagicMock(total_size=100, num_pieces=1, pieces=[piece]),
            bitfield=bf,
            peers=[],
        )

        mock_response = MagicMock()
        mock_response.content = b"\xAB" * 100  # won't match "expected_but_wrong"
        mock_response.raise_for_status = MagicMock()

        with patch("tightwad.swarm_transfer.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_httpx.AsyncClient.return_value = mock_client

            sem = __import__("asyncio").Semaphore(4)
            ok = await puller.download_piece(piece, peer, sem)

        assert ok is False
        assert 0 not in bf.have
