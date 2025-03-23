<h2 align='center'>
    SmartHome-Bench
</h2>

<h4 align='center'>
SmartHome-Bench: A Comprehensive Benchmark for Video Anomaly Detection in Smart Homes Using Multi-Modal Large Language Models
</h4>

[![UWISE](https://img.shields.io/badge/About-UW/ISE-blue.svg)](https://ise.washington.edu/)
[![WYZE](https://img.shields.io/badge/About-WYZE-orange)](https://www.wyze.com/)

<p align="center">
 <img src="Figures/video_category_distribution.png" width="80%">
    <br>
    <em>Fig. 1 Distribution of Video Anomaly Tags and Taxonomy Categories.</em>
</p>




## :star: Features

**SmartHome-Bench** is the first video dataset specifically designed for smart home surveillance scenarios. It serves as a benchmark not only for video anomaly detection (VAD) but also offers a versatile evaluation platform to comprehensively assess the performance of multi-modal large language models (MLLMs) in video reasoning and interpretation. Please see our paper here: (add link).

-  :house: First benchmark specifically designed for detecting anomalies in smart home videos.
- :movie_camera: Dataset includes 1,203 video clips, each annotated with:
  - Detailed video description; 
  - Anomaly or normality reasoning;
  - Binary anomaly tag (0 for normal, 1 for abnormal)
- :triangular_ruler: Introduction of the first video anomaly taxonomy, covering seven common scenario categories.
- :fire: Evaluation of state-of-the-art closed-source and open-source LLMs using various prompting techniques.


<!--
##   :books: Anomaly Taxonomy
-->
<p align="center">
 <img src="Figures/taxonomy.png" width="60%">
    <br>
    <em>Fig. 2 Overview of the video anomaly taxonomy in smart homes.</em>
</p>



>Figure 2 provides an overview of this taxonomy, which covers seven categories of scenarios that frequently occur at home and are of concern to users. Each category is further divided into normal and abnormal videos, with detailed descriptions provided for both.
<!--
>For instance, the taxonomy for the **Wildlife** category includes the following situation for normal videos and abnormal videos:
-->
<!--
- **Normal Videos**:
  - **Harmless Wildlife**: Harmless wildlife sightings, such as squirrels, birds, or rabbits, moving through the yard.
  - **Common Pests**: Common pest activity that doesn’t pose immediate danger (e.g., bugs in the garden).

- **Abnormal Videos**:
  - **Dangerous Wildlife**: Presence of dangerous wildlife like snakes, spiders, or raccoons that may pose a health risk.
  - **Wildlife Damage**: Any wildlife activity that causes or potentially causes damage to property or threatens human or pet safety.
  - **Indoor Wildlife**: Any wildlife (dangerous or not) that enters a home without clear containment.

-->


## :minidisc: Video Collection

1. Videos were collected primarily from public resources, such as YouTube, based on seven taxonomy categories.
2. Specific keywords were developed for each category to guide the search.
   - Example: "cat play home cam" for normal pet monitoring footage, and "pet vomit home cam" for abnormal events.
3. Videos were scraped using these keywords and carefully screened to ensure they were exclusively captured by smart home cameras.


<!--
##   :bar_chart: Dataset Statistics

Our **SmartHome-Bench** dataset consists of 1,203 smart home video clips across 7 categories. As shown in Figure 1, the dataset is balanced, with a similar number of abnormal and normal videos. Among the 7 categories, the security category contains the most videos.
-->


<p align="center">
 <img src="Figures/video_time_distribution.png">
    <br>
    <em>Fig. 3 Distribution of Video Duration, Description Word Count, and Reasoning Word Count.</em>
</p>

##  

>Figure 3 presents the distribution of the time durations for these 1,203 video clips, as well as the word count distribution for the descriptions and reasoning annotations, offering insights into the complexity of the videos.

- The majority of smart home video clips are shorter than 80 seconds.
- Reasoning annotations are typically more concise than descriptions, as they focus only on the key event leading to the anomaly tag.
- Descriptions provide a more detailed account of all events in the video.


## :wrench: How to Use
### Downaloading Videos
1. The video URLs are provided in an Excel file [Video_url.xlsx](https://github.com/Xinyi-0724/SmartHome-Bench-LLM/tree/main/Videos/Video_url.xlsx) of this GitHub repository. The first 1,023 videos can be downloaded from YouTube using the provided URLs, while the remaining 180 videos, contributed by our staff, are private and cannot be downloaded.

2. After downloading the videos, you need to trim them to extract the specific clips used in our paper. This can be done using the script [Video_trim.ipynb](https://github.com/Xinyi-0724/SmartHome-Bench-LLM/blob/main/Videos/Trim_Videos/Video_trim.ipynb) and the trim time information available in [Trim_video_label.csv](https://github.com/Xinyi-0724/SmartHome-Bench-LLM/blob/main/Videos/Trim_Videos/Trim_video_label.csv).

3. The complete video annotation details for all 1,203 videos can be found in [Video_Annotation.xlsx](https://github.com/Xinyi-0724/SmartHome-Bench/blob/main/Videos/Video_Annotation.xlsx).

>   **Acknowledgment:** We thank Kevin Beussman for donating the videos. We also thank Pengfei Gao, Xiaoya Hu, Liting Jia, Lina Liu, Vincent Nguyen, and Yunyun Xi for assisting with the video annotation process.


### Running Models and Evaluation
To run our experiments, you can select:

- **Model:** ('Claude', 'Flash', 'GPT', 'GPTmini', 'Pro', 'VILA')
- **Method:** ('zeroshot', 'fewshot', 'COT')
- **Step:** (`Step1` for generating model responses, `Step2` for calculating accuracy. VILA only has Step2.)

After downloading this repository, run the following command:

```bash
python run.py --model <model_name> --method <method_name> --step <Step1 or Step2>
```



</div>



## :smiley: Citing 

If you use **SmartHome-Bench** in a scientific publication, we would appreciate citations to:

```bibtex
@article{zhao2024smarthome,
  title={SmartHome-Bench: A Comprehensive Benchmark for Video Anomaly Detection in Smart Homes Using Multi-Modal Large Language Models},
  author={Zhao, Xinyi and Zhang, Congjing},
  journal={arXiv preprint arXiv: },
  year={2024},
}
```

