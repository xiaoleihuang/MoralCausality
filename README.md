# Examining Variations of Moral Values across Social Movements HT - 2022
Data and code repository to for the HT 2022 paper [Learning to Adapt Domain Shifts of Moral Values via Instance Weighting](https://dl.acm.org/doi/10.1145/3511095.3531269).


# Data
We have developed and annotated moral values of COVID-19 vaccine hesitancy in this study.
	* The data resources of COVID-19 vaccine in this link: https://zenodo.org/record/6508958#.YrS0QuzMJhE


# How to Run
* Install the following:;
  * Install [conda](https://www.anaconda.com/distribution/);
  * Please refer to model folder to run the jupyter notebooks.


# Presentation and Slides
You can find my presentation files under the `resources` directory.

# Contact
Please email **xiaolei.huang@memphis.edu** for further discussion. Feel free to open an issue regarding running guidelines. **Note that I don't have much time to re-organize the code due to funding proposals and other projects...**


# Citation
If you use our corpus in your publication, please kindly cite this [paper](https://dl.acm.org/doi/10.1145/3511095.3531269):

```
@inproceedings{huang2022learning,
	author = {Huang, Xiaolei and Wormley, Alexandra and Cohen, Adam},
	title = {Learning to Adapt Domain Shifts of Moral Values via Instance Weighting},
	year = {2022},
	isbn = {9781450392334},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3511095.3531269},
	doi = {10.1145/3511095.3531269},
	abstract = {Classifying moral values in user-generated text from social media is critical in understanding community cultures and interpreting user behaviors of social movements. Moral values and language usage can change across the social movements; however, text classifiers are usually trained in source domains of existing social movements and tested in target domains of new social issues without considering the variations. In this study, we examine domain shifts of moral values and language usage, quantify the effects of domain shifts on the morality classification task, and propose a neural adaptation framework via instance weighting to improve cross-domain classification tasks. The quantification analysis suggests a strong correlation between morality shifts, language usage, and classification performance. We evaluate the neural adaptation framework on a public Twitter data across 7 social movements and gain classification improvements up to 12.1%. Finally, we release a new data of the COVID-19 vaccine labeled with moral values and evaluate our approach on the new target domain. For the case study of the COVID-19 vaccine, our adaptation framework achieves up to 5.26% improvements over neural baselines. This is the first study to quantify impacts of moral shifts, propose adaptive framework to model the shifts, and conduct a case study to model COVID-19 vaccine-related behaviors from moral values.},
	booktitle = {Proceedings of the 33rd ACM Conference on Hypertext and Social Media},
	pages = {121–131},
	numpages = {11},
	keywords = {moral values, adaptation, morality, classification, instant weighting, domain variation},
	location = {Barcelona, Spain},
	series = {HT '22}
}
```
