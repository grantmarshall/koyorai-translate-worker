import argparse
import io
import numpy
import soundfile as sf
import sox
import sqlite3
import whisper


def get_oldest_session(db):
    return None


def get_bytes_for_session(db, id):
    chunks = [bytes(chunk[0]) for chunk in db.execute(
        'SELECT c.chunk AS chunk FROM audio_chunks c'
        ' WHERE c.session_id = ?'
        ' ORDER BY c.user_ts ASC',
        (id, )
    ).fetchall()]
    print('Found {} chunks in session {}'.format(len(chunks), id))
    return sf.read(io.BytesIO(b''.join(chunks)))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-db', '--database', type=str)
    parser.add_argument('-s', '--session', type=str)
    args, _ = parser.parse_known_args()
    db = sqlite3.connect(
        args.database,
        detect_types=sqlite3.PARSE_DECLTYPES
    )

    # Create a model to be reused for translation
    print('Loading model...')
    model = whisper.load_model("large")
    print('Model loaded.')

    # Create an SOX downsampler to convert 48KHz data to 16KHz
    tfm = sox.Transformer()
    tfm.set_output_format(channels=1, rate=16000)

    # TODO(grantmarshall) Get the most out of date session from the DB
    # oldest_session_id = get_oldest_session(db)

    # Get raw bytes for the audio of the session as a numpy array
    data, sample_rate = get_bytes_for_session(db, args.session)

    # Downsample the data and feed it into the translator
    downsampled_data = tfm.build_array(
        input_array=data,
        sample_rate_in=sample_rate
    )
    result = whisper.transcribe(
        audio=downsampled_data.astype(numpy.float32),
        model=model,
        task="translate")

    # TODO(grantmarshall): Insert the translation into the db
    print(result)


if __name__ == '__main__':
    main()
