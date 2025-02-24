# Tech Stack

## Collecting and Analyzing the Dataset
1. Use Bash to scrape data from [www.crossfireforum.org](https://www.crossfireforum.org).  
2. Download relevant PDFs associated with the vehicle.  
3. Use [AugmentedToolKit](https://github.com/e-p-armstrong/augmentoolkit) to generate synthetic datasets from the scraped data.  
4. Perform topic modeling to better understand the dataset.  

## Fine-Tuning the Model
1. Use [Unsloth](https://github.com/unslothai/unsloth) Unsloth to fine-tune a LLaMA-based LLM.  
2. Use WandB to optimize and sweep training parameters.  
3. Use [Microsoft's PromptWizard](https://github.com/microsoft/PromptWizard) to generate system prompts at inference.  
4. Optimize inference parameters to improve model performance.  
5. Develop methods to evaluate the model’s performance.  

## Deploying the AI Agent
1. Identify a suitable platform for deploying the model as a chatable AI Agent.  
