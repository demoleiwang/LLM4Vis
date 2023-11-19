# LLM4Vis
Code for Our EMNLP (Industry) 2023 paper "LLM4Vis: Explainable Visualization Recommendation using ChatGPT"

### Run in-context learning for visualization recommendation.

Unzip the dataset file example2_nodiscrization_3.csv.zip.

Set an api-key of OpenAI API in the utils file.
```shell
openai.api_key = ""
openai.api_base = "" 
```
Since we have prepared all the relevant files, you can directly run the following command.
```shell
python final_run.py
```
Check our result log file (result_file.log) in the output directory.


### Run to get relevant files.

Run the following command to get summary file
```shell
python feature_summary.py
```

Run the following command to get demonstration file, including the code for explanation generation bootstrapping.
```shell
python demo_prepare.py
```

Run the following command to get similarity file
```shell
python similarity.py
```

## :smile_cat: Cite

If you find **LLM4Vis** useful for your research and applications, please kindly cite using this BibTeX:

```latex
@article{wang2023llm4vis,
  title={LLM4Vis: Explainable Visualization Recommendation using ChatGPT},
  author={Wang, Lei and Zhang, Songheng and Wang, Yun and Lim, Ee-Peng and Wang, Yong},
  journal={arXiv preprint arXiv:2310.07652},
  year={2023}
}
```

