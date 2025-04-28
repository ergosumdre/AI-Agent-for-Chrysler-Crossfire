library(rvest)
library(dplyr)
library(stringr)

clean_forum_post = function(text) {
  text = str_replace_all(text, "\\{\"@context\".*?\\}", "")
  text = str_replace_all(text, "vbmenu_register\\(.*?\\);", "")
  text = str_replace_all(text, "Share Options", "")
  text = str_replace_all(text, "#\\d+\\s*\\(permalink\\)", "")
  text = str_replace_all(text, "Joined:.*?Likes:", "")
  text = str_replace_all(text, "Posts:.*?From:", "")
  text = str_replace_all(text, "Quote:\\s*Originally Posted by.*?:", "")
  text = str_replace_all(text, "\\w{3} \\d{1,2}, \\d{4} \\| \\d{2}:\\d{2} [AP]M", "")
  text = str_squish(text)
  text = str_trim(text)
  text = ifelse(text == "" | text == "0", NA, text)
  return(text)
}

# Function to parse HTML files from subdirectories
parse_forum_posts = function(base_directory) {
  # List of subdirectories to process
  subdirectories = c(
    'all-crossfires', 
    'crossfire-coupe', 
    'crossfire-roadster', 
    'crossfire-srt6', 
    'troubleshooting-technical-questions-modifications', 
    'tsbs-how-articles', 
    'wheels-brakes-tires-suspension'
  )
  
  forum_posts_list = list()
  
  for (subdir in subdirectories) {
    full_path = file.path(base_directory, subdir)
    
    html_files = list.files(path = full_path, pattern = "\\.html$", full.names = TRUE)
    html_files = html_files[!grepl("/index\\.html$", html_files, ignore.case = TRUE)]
    
    # Process each HTML file
    for (file in html_files) {
      tryCatch({
        html_content = read_html(file)
        posts = html_content %>%
          html_nodes("div.tpost") %>%  # Target content div
          html_text(trim = TRUE) %>% 
          str_replace_all("\\s+", " ") %>%     # Normalize whitespace
          str_replace_all("[\r\n]+", "\n")     # Standardize line breaks
        
        cleaned_posts = sapply(posts, clean_forum_post)
        cleaned_posts = cleaned_posts[!is.na(cleaned_posts)]
        combined_posts = paste(cleaned_posts, collapse = " || ")
        webpage_entry = data.frame(
          subdirectory = subdir,
          filename = basename(file),
          post_content = combined_posts,
          stringsAsFactors = FALSE
        )
        
        forum_posts_list[[file]] = webpage_entry
      }, error = function(e) {
        cat("Error processing file:", file, "\n")
        cat("Error message:", conditionMessage(e), "\n")
      })
    }
  }
  forum_posts_df = do.call(rbind, forum_posts_list)
  return(forum_posts_df)
}

# Set base directory
base_directory = "/Users/dre/Downloads/crossfire_scrape/www.crossfireforum.org/forum"

# Execute parsing
dat = parse_forum_posts(base_directory)

text_corpus = dat$post_content
writeLines(text_corpus, "atk_cf.txt")
