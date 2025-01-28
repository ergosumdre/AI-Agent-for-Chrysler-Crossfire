# Tech Stack

## Collecting/Analyzing the Dataset
#### 1. Use bash to scrape the website www.crossfireforum.org
#### 2. Download relevant PDF associated to the vehicle. 
#### 3. Use AugmentedToolKit to generate synthetic datasets from the scraped data. 
#### 4. Perform Topic Modeling to better understand the dataset.

## Fine-tuning the Model
#### 1. Use Unsloth to fine-tune a llama based LLM. 
#### 2. Use WandB to optimize/sweep training parameters.
#### 3. Use Microsoft's PromptWizard to generate system prompts at inference.
#### 4. Optimize inference parameters to improve the model's performance.
#### 5. Find ways to evaluate the Models performance.

## Deploying the Model/Chatbot
#### 1. Find a platform to deploy the model as a chatbot.
