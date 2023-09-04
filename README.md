# mental_health_monitoring_PIPELINE
Description of Androids-Corpus

• The distinction between depressed and non-depressed speak-
ers was made by professional psychiatrists and not through
the administration of self-assessment questionnaires. This
means that the data allows one to investigate depression de-
tection and not prediction of scores obtained through ques-
tionnaires, known to be affected by multiple biases [17].

Pipeline for Androids-Corpora

1- Create label.txt with groundtruth. Its structure should be as follows:

    ID  Gender  Age  Educational-level  Condition
    
    *   *       *    *                   *
     
    ID: Speaker ID
    Condition: mental health state. Binary representation. For example: AD/non-AD

2- Run analysis_audio.py script to analize Corpus

3- Gathering audios withour interviewer

4- Run feature_extraction.py script
  * Features are extracted by segments of 1 minutes. It avoids running out of memory and also gives a context to the normalization strategy
  * Analyse result with and without apply VAD and the influence of the sequence length in this study

5- Run folds_default_Androids-Corpus.py for creating default fold list per tasks


