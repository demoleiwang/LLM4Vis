

template_icl_sum = """
The features of a given tabular dataset are provided in the following delimited by triple backticks. \
Your task is to generate a detailed text description, in 1000 characters, that focus on features that are \
important for visualization type selection and \
comprehensively analyzes this tabuar dataset based on its feature values from both single-column and cross-column perspectives. \
Note that the response must exclude words such as line chart, scatter plot, bar chart, and box plot, since \
these words will mislead further visualization recommendation. \
The response format can be as "Single-column perspective: [...]. \n\nCross-column perspective: [...]."
Ensure that the summary maintains strong generalization ability and includes all vital information.

Features for a tabular dataset: ```{}```

"""

role_str_sum = """
You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.
Knowledge cutoff: September 2021
Current date: July 7, 2023

As a visualization recommender, your task is to select the most appropriate visualization \
types to represent tabular data using line charts, scatter plots, bar charts, and box plots. \
Based on your knowledge of data visualization and the given rules and features of the data, \
recommend the best visualization labels that effectively communicate the data insights to the \
target audience. Your recommendations should consider the data distribution, relationship 、
between variables, and the purpose of the visualization.

"""



role_str_demo_prepare = """
You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.
Knowledge cutoff: September 2021
Current date: July 7, 2023
"""


demos_descri="""
====
Determine whether each visualization type in the following \
list of visualization types is a suitable visualization type \
in the text description for a tabular dataset below, which \
is delimited with triple backticks. \

Give your explanation and your answer at the end as json \
(Explanation is as below: .\n The final answer in JSON format would be:), \
where each element consists of a visualization type and \
a score ranging from 0 to 1 (1 means the most suitable).  
The scores should sum to be 1 (line + scatter + bar + box = 1.0). \

List of visualization types: [line chart, scatter plot, bar chart, and box plot]. \

Text description for a tabular dataset: ```{}```

"""

demos_descri_hint="""
====
Determine whether each visualization type in the following \
list of visualization types is a suitable visualization type \
in the text description for a tabular dataset below, which \
is delimited with triple backticks. \

Hint: {} may be more suitable than {}, however previous score is {}. \

With the given hint, \
editing your explanation and improve your answer at the end as json \
(Explanation is as below: .\n The final answer in JSON format would be:), \
where each element consists of a visualization type and \
a score ranging from 0 to 1 (1 means the most suitable). 
The scores should sum to be 1 (line + scatter + bar + box = 1.0). \

List of visualization types: [line chart, scatter plot, bar chart, and box plot]. \

Text description for a tabular dataset: ```{}```

"""

demos_descri_x="""
====
Determine whether each visualization type in the following \
list of visualization types is a suitable visualization type \
in the text description for a tabular dataset below, which \
is delimited with triple backticks. \

Give your explanation and your answer at the end as json \
(Explanation is as below: .\n The final answer in JSON format would be:), \
where each element consists of a visualization type and \
a score ranging from 0 to 1 (1 means the most suitable). \ 
The scores should sum to be 1 (line + scatter + bar + box = 1.0). \

List of visualization types: [line chart, scatter plot, bar chart, and box plot]. \

Text description for a tabular dataset: ```{}```
"""




