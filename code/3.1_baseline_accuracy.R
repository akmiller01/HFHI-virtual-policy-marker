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
  # "input/accuracy_4o_20250206.csv",
  # "input/accuracy_phi4_20250206.csv",
  # "input/accuracy_phi4_20250207.csv",
  # "input/accuracy_20250207_nosector.csv",
  # "input/accuracy_20250207_nosector2.csv",
  # "input/accuracy_20250207_nosector2_lowtemp.csv",
  # "input/accuracy_20250207_strict_lowtemp.csv",
  # "input/accuracy_20250207_lessstrict_midtemp.csv",
  # "input/accuracy_20250207_lessstrict.csv"
  # "input/accuracy_20250208.csv",
  # "input/accuracy_20250208_lambda_1.csv",
  # "input/accuracy_20250208_lambda_2.csv",
  # "input/accuracy_20250208_lambda_3.csv",
  # "input/accuracy_20250208_lambda_4.csv",
  # "input/accuracy_20250208_lambda_5.csv",
  "input/accuracy_20250208_sector_1.csv",
  "input/accuracy_20250208_sector_2.csv",
  "input/accuracy_20250208_sector_3.csv",
  # "input/accuracy_20250208_sector_zerotemp_1.csv",
  # "input/accuracy_20250208_sector_zerotemp_2.csv",
  # "input/accuracy_20250208_sector_zerotemp_3.csv",
  # "input/accuracy_20250208_sector_04temp_1.csv",
  # "input/accuracy_20250208_sector_04temp_2.csv",
  # "input/accuracy_20250208_sector_04temp_3.csv"
  # "input/accuracy_20250208_sector_listthoughts_1.csv",
  # "input/accuracy_20250208_sector_listthoughts_2.csv",
  # "input/accuracy_20250208_sector_listthoughts_3.csv"
  "input/accuracy_20250208_corrections_1.csv",
  "input/accuracy_20250208_corrections_2.csv",
  "input/accuracy_20250208_corrections_3.csv"
  
)
keys = c(
  "Housing", "Homelessness",
  "Transitional", "Incremental",
  "Social", "Market", "Urban", "Rural"
)
accuracy_list = list()
for(data_file in data_files){
  dat = fread(data_file)
  all_actual = c()
  all_pred = c()
  tmp_df = data.frame(name=basename(data_file))
  for(key in keys){
    ai_col = paste(key,"AI")
    ai_values = dat[,ai_col,with=F]
    all_pred = c(all_pred, ai_values[[1]])
    manual_col = paste(key, "DK")
    manual_values = dat[,manual_col,with=F]
    all_actual = c(all_actual, manual_values[[1]])
    tmp_df[,paste(key,"accuracy",sep="_")] = mean(manual_values==ai_values)
    tmp_df[,paste(key,"precision",sep="_")] = precision(manual_values, ai_values)
    tmp_df[,paste(key,"recall",sep="_")] = recall(manual_values, ai_values)
    tmp_df[,paste(key,"f1",sep="_")] = f1_score(manual_values, ai_values)
  }
  tmp_df[,"overall_accuracy"] = mean(all_actual==all_pred)
  tmp_df[,"overall_precision"] = precision(all_actual, all_pred)
  tmp_df[,"overall_recall"] = recall(all_actual, all_pred)
  tmp_df[,"overall_f1"] = f1_score(all_actual, all_pred)
  accuracy_list[[data_file]] = tmp_df
}

metrics = rbindlist(accuracy_list)
fwrite(metrics, "output/metrics.csv")
# metrics_long = melt(metrics, id.vars=c("name"))
# metrics_long$variable = as.character(metrics_long$variable)
# metrics_long$indicator = sapply(strsplit(metrics_long$variable, split="_"), `[[`, 1)
# metrics_long$metric = sapply(strsplit(metrics_long$variable, split="_"), `[[`, 2)
# metrics_wide = dcast(metrics_long, indicator+metric~name)
# fwrite(metrics_wide, "output/metrics.csv")