# Train with share=0.0

python -u train_ts2vecar.py SelfRegulationSCP2 default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py StandWalkJump default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py SpokenArabicDigits default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py DuckDuckGeese default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py ArticularyWordRecognition default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py CharacterTrajectories default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py EigenWorms default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py PenDigits default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --time_steps 4 --eval
python -u train_ts2vecar.py Handwriting default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py NATOPS default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py RacketSports default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval
python -u train_ts2vecar.py UWaveGestureLibrary default_ar_1 --loader UEA --max-threads 8 --seed 42 --share 0.0 --eval

# Train with share=0.1

python -u train_ts2vecar.py SelfRegulationSCP2 default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py StandWalkJump default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py SpokenArabicDigits default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py DuckDuckGeese default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py ArticularyWordRecognition default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py CharacterTrajectories default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py EigenWorms default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py PenDigits default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --time_steps 4 --eval
python -u train_ts2vecar.py Handwriting default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py NATOPS default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py RacketSports default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval
python -u train_ts2vecar.py UWaveGestureLibrary default_ar_2 --loader UEA --max-threads 8 --seed 42 --share 0.1 --eval

# Train with share=0.2

# python -u train_ts2vecar.py SelfRegulationSCP2 default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py StandWalkJump default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py SpokenArabicDigits default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py DuckDuckGeese default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py ArticularyWordRecognition default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py CharacterTrajectories default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py EigenWorms default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py PenDigits default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --time_steps 4 --eval
# python -u train_ts2vecar.py Handwriting default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py NATOPS default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py RacketSports default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval
# python -u train_ts2vecar.py UWaveGestureLibrary default_ar_3 --loader UEA --max-threads 8 --seed 42 --share 0.2 --eval

# Replicate original model

python -u train_ts2vec.py SelfRegulationSCP2 replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py StandWalkJump replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py SpokenArabicDigits replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py DuckDuckGeese replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py ArticularyWordRecognition replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py CharacterTrajectories replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py EigenWorms replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py PenDigits replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py Handwriting replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py NATOPS replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py RacketSports replicated --loader UEA --max-threads 8 --seed 42 --eval
python -u train_ts2vec.py UWaveGestureLibrary replicated --loader UEA --max-threads 8 --seed 42 --eval
