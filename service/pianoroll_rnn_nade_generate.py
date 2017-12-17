import ast
import os
import time

# internal imports

import tensorflow as tf
import magenta

from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_model
from magenta.models.pianoroll_rnn_nade.pianoroll_rnn_nade_sequence_generator import PianorollRnnNadeSequenceGenerator

from magenta.music import constants
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2
from flask import request, send_file, jsonify, Flask

app = Flask(__name__, static_folder = "tmp")
app.config.update(
    WTF_CSRF_ENABLED = True,
    SECRET_KEY = 'you-will-never-guess'
)

bundle_file = 'pretrained/pianoroll_rnn_nade.mag'
output_dir = '/tmp/pianoroll_rnn_nade/generated'
num_outputs = 1
num_steps = 128
qpm = None
beam_size = 1
branch_factor = 1
log = 'INFO'
hparams = ''


def get_bundle():
    global bundle_file
    bundle_file = os.path.expanduser(bundle_file)
    return magenta.music.read_bundle_file(bundle_file)


def generate_from_primer(generator, primer_pitches=None):

    global output_dir, qpm
    output_dir = os.path.expanduser(output_dir)

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    qpm = qpm if qpm else 60
    if primer_pitches:
        primer_sequence = music_pb2.NoteSequence()
        primer_sequence.tempos.add().qpm = qpm
        primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
        for pitch in ast.literal_eval(primer_pitches):
            note = primer_sequence.notes.add()
            note.start_time = 0
            note.end_time = 60.0 / qpm
            note.pitch = pitch
            note.velocity = 100
        primer_sequence.total_time = primer_sequence.notes[-1].end_time

    else:
        tf.logging.warning(
            'No priming sequence specified. Defaulting to empty sequence.')
        primer_sequence = music_pb2.NoteSequence()
        primer_sequence.tempos.add().qpm = qpm
        primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ

    # Derive the total number of seconds to generate.
    seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
    generate_end_time = num_steps * seconds_per_step

    generator_options = generator_pb2.GeneratorOptions()
    # Set the start time to begin when the last note ends.
    generate_section = generator_options.generate_sections.add(
        start_time=primer_sequence.total_time,
        end_time=generate_end_time)

    if generate_section.start_time >= generate_section.end_time:
        tf.logging.fatal(
            'Priming sequence is longer than the total number of steps '
            'requested: Priming sequence length: %s, Total length '
            'requested: %s',
            generate_section.start_time, generate_end_time)
        return

    generator_options.args['beam_size'].int_value = beam_size
    generator_options.args['branch_factor'].int_value = branch_factor

    # tf.logging.info('primer_sequence: %s', primer_sequence)
    # tf.logging.info('generator_options: %s', generator_options)

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(num_outputs))
    midi_path = None
    for i in range(num_outputs):
        generated_sequence = generator.generate(primer_sequence, generator_options)

        midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
        midi_path = os.path.join(output_dir, midi_filename)
        # print(generated_sequence)
        magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

    return midi_path


@app.route('/api/v0/generate', methods=['GET'])
def gen_api():
    primer = request.args.get('primer', type=str)
    print(primer)
    midi_path = generate_from_primer(generator, primer_pitches=primer)
    filename = midi_path.split('/')[-1].split('\\')[-1].split('\\\\')[-1]
    if os.path.isfile(midi_path):
        try:
            result_file = os.path.abspath(midi_path)
            return send_file(result_file, attachment_filename=filename, as_attachment=True)
        except:
            pass
    message = {
        'status': 400,
        'message': "File not found",
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp


if __name__ == '__main__':
    bundle_file = 'pretrained/pianoroll_rnn_nade.mag'
    with tf.Session():
        tf.logging.set_verbosity(log)

        bundle = get_bundle()

        config_id = bundle.generator_details.id
        config = pianoroll_rnn_nade_model.default_configs[config_id]
        config.hparams.parse(hparams)
        config.hparams.batch_size = min(
            config.hparams.batch_size, beam_size * branch_factor)

        generator = PianorollRnnNadeSequenceGenerator(
            model=pianoroll_rnn_nade_model.PianorollRnnNadeModel(config),
            details=config.details,
            steps_per_quarter=config.steps_per_quarter,
            checkpoint=None,
            bundle=bundle)
        app.run(debug=True, host='0.0.0.0', use_reloader=False)
