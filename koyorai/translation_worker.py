import argparse
import io
import numpy
import soundfile as sf
import sox
import sqlite3
import time
import whisper


def get_oldest_session(db):
    print('Getting oldest session.')
    ids = [row[0] for row in db.execute(
        'SELECT c.session_id AS session_id FROM audio_chunks c'
        ' ORDER BY c.insert_ts DESC'
    ).fetchall()]
    return ids[0]


def get_bytes_for_session(db, id):
    chunks = []
    last_chunk_id = -1
    for chunk in db.execute(
        'SELECT c.chunk AS chunk, c.id AS audio_chunk_id FROM audio_chunks c'
        ' WHERE c.session_id = ?'
        ' ORDER BY c.user_ts ASC',
        (id, )
    ).fetchall():
        chunks.append(bytes(chunk[0]))
        last_chunk_id = int(chunk[1])

    print('Found {} chunks in session {} with last chunk id {}'
          .format(len(chunks), id, last_chunk_id))
    return last_chunk_id, b''.join(chunks)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-db', '--database', type=str)
    args, _ = parser.parse_known_args()
    db = sqlite3.connect(args.database, detect_types=sqlite3.PARSE_DECLTYPES)

    # Create a model to be reused for translation
    print('Loading model...')
    model = whisper.load_model("large")
    print('Model loaded.')

    # Create an SOX downsampler to convert 48KHz data to 16KHz
    tfm = sox.Transformer()
    tfm.set_output_format(channels=1, rate=16000)
    times = []

    # Get raw bytes for the audio of the session as a numpy array
    while True:
        session_id = get_oldest_session(db)
        last_chunk_id, session_bytes = get_bytes_for_session(db, session_id)
        data, sample_rate = sf.read(io.BytesIO(session_bytes))

        # Downsample the data and feed it into the translator
        print('Beginning translation.')
        begin_time = time.time()
        downsampled_data = tfm.build_array(
            input_array=data[-30 * sample_rate:],
            sample_rate_in=sample_rate
        )
        result = whisper.transcribe(
            audio=downsampled_data.astype(numpy.float32),
            model=model,
            task="translate")
        elapsed_time = (time.time() - begin_time)
        times.append(elapsed_time)

        print('Computation time: {:03.2f}s'.format(elapsed_time))
        print('Running average: {:03.2f}s'.format(sum(times) / len(times)))
        print('Result: {}'.format(result['text']))
        db.execute('INSERT INTO translations (session_id, audio_chunk_id, tl)'
                   ' VALUES (?, ?, ?)',
                   (session_id, last_chunk_id, result['text']))
        db.commit()


if __name__ == '__main__':
    main()
