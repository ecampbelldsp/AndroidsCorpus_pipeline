# mental_health_monitoring_PIPELINE

Pipeline

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
