import tempfile
import unittest
from pathlib import Path

from storage import AudioArtifactStore


class AudioArtifactStoreTests(unittest.TestCase):
    def test_create_audio_file_uses_unique_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AudioArtifactStore(Path(temp_dir))

            first_path = store.create_audio_file(
                b"hello-audio",
                mime_type="audio/wav",
                file_name="reply.wav",
            )
            second_path = store.create_audio_file(
                b"hello-audio-again",
                mime_type="audio/wav",
                file_name="reply.wav",
            )

            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertNotEqual(first_path.name, second_path.name)
            self.assertEqual(first_path.read_bytes(), b"hello-audio")
            self.assertEqual(second_path.read_bytes(), b"hello-audio-again")

    def test_delete_removes_audio_file_from_managed_spool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AudioArtifactStore(Path(temp_dir))
            audio_path = store.create_audio_file(
                b"hello-audio",
                mime_type="audio/wav",
                file_name="reply.wav",
            )

            store.delete(audio_path)

            self.assertFalse(audio_path.exists())

    def test_resolve_managed_path_rejects_paths_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AudioArtifactStore(Path(temp_dir))

            with self.assertRaises(ValueError):
                store.resolve_managed_path(Path(temp_dir).parent / "other.wav")


if __name__ == "__main__":
    unittest.main()
