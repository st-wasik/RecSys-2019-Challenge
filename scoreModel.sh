python3 2019/src/verify_submission/verify_subm.py --data-path splitterOut/
#python3 2019/src/score_submission/score_subm.py --submission-file submissionOUT.csv --ground-truth-file splitterOut/gt.csv --data-path .
python3 2019/src/score_submission/score_subm.py --submission-file splitterOut/model.csv --ground-truth-file splitterOut/gt.csv --data-path .
