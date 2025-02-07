#### Setup ####
list.of.packages <- c("data.table", "rstudioapi", "scales","ggplot2","scales","Hmisc","openxlsx", "countrycode", "furrr", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd <- dirname(getActiveDocumentContext()$path) 
setwd(wd)
setwd("../")

# Function to calculate precision
precision <- function(true_labels, predicted_labels) {
  tp <- sum(true_labels & predicted_labels)  # True Positives
  fp <- sum(!true_labels & predicted_labels) # False Positives
  if ((tp + fp) == 0) return(0)  # Avoid division by zero
  return(tp / (tp + fp))
}

# Function to calculate recall
recall <- function(true_labels, predicted_labels) {
  tp <- sum(true_labels & predicted_labels)  # True Positives
  fn <- sum(true_labels & !predicted_labels) # False Negatives
  if ((tp + fn) == 0) return(0)  # Avoid division by zero
  return(tp / (tp + fn))
}

# Function to calculate F1-score
f1_score <- function(true_labels, predicted_labels) {
  p <- precision(true_labels, predicted_labels)
  r <- recall(true_labels, predicted_labels)
  
  if ((p + r) == 0) return(0)  # Avoid division by zero
  return(2 * p * r / (p + r))
}

data_files = c(
  "input/manually_coded_for_accuracy.csv",
  "input/accuracy_4o_20250206.csv",
  "input/accuracy_phi4_20250206.csv",
  "input/accuracy_phi4_20250207.csv"
)
keys = c(
  "Housing", "Homelessness",
  "Transitional", "Incremental",
  "Social", "Market", "Urban", "Rural"
)
accuracy_list = list()
for(data_file in data_files){
  dat = fread(data_file)
  tmp_df = data.frame(name=basename(data_file))
  for(key in keys){
    ai_col = paste(key,"AI")
    ai_values = dat[,ai_col,with=F]
    manual_col = paste(key, "DK")
    manual_values = dat[,manual_col,with=F]
    tmp_df[,paste(key,"accuracy",sep="_")] = mean(manual_values==ai_values)
    tmp_df[,paste(key,"precision",sep="_")] = precision(manual_values, ai_values)
    tmp_df[,paste(key,"recall",sep="_")] = recall(manual_values, ai_values)
    tmp_df[,paste(key,"f1",sep="_")] = f1_score(manual_values, ai_values)
  }
  accuracy_list[[data_file]] = tmp_df
}

metrics = rbindlist(accuracy_list)
metrics_long = melt(metrics, id.vars=c("name"))
metrics_long$variable = as.character(metrics_long$variable)
metrics_long$indicator = sapply(strsplit(metrics_long$variable, split="_"), `[[`, 1)
metrics_long$metric = sapply(strsplit(metrics_long$variable, split="_"), `[[`, 2)
metrics_wide = dcast(metrics_long, indicator+metric~name)
fwrite(metrics_wide, "output/metrics.csv")
