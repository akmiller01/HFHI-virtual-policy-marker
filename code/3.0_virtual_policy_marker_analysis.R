#### Setup ####
list.of.packages <- c("data.table", "rstudioapi", "scales","ggplot2","scales","Hmisc","openxlsx", "countrycode", "furrr", "stringr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only=T)

wd <- dirname(getActiveDocumentContext()$path) 
setwd(wd)
setwd("../")
#### End setup ####

#### Chart setup ####

master_blue = "#005596"
master_green = "#54B948"
master_black = "#000000"

primary_red = "#DC241F"
primary_yellow = "#F1AB00"
primary_orange = "#9A3416"
primary_purple = "#4A207E"

secondary_blue = "#00759F"
secondary_green = "#007B63"
secondary_yellow = "#C69200"
secondary_orange = "#DA5C05"

tertiary_blue = "#93B9DC"
tertiary_grey = "#7C7369"
tertiary_tan = "#D3BE96"

six_pallette = c(
  primary_orange, primary_yellow, master_green, master_blue, primary_purple, tertiary_grey
)

custom_style = theme_bw() +
  theme(
    panel.border = element_blank()
    ,panel.grid.major.x = element_blank()
    ,panel.grid.minor.x = element_blank()
    ,panel.grid.major.y = element_line(colour = "#a9a6aa")
    ,panel.grid.minor.y = element_blank()
    ,panel.background = element_blank()
    ,axis.line.x = element_line(colour = "black")
    ,axis.line.y = element_blank()
    ,axis.ticks = element_blank()
    ,legend.position = "bottom"
  )

donut_style = custom_style +
  theme(
    panel.grid.major.y = element_blank()
    ,axis.line.x = element_blank()
    ,axis.text = element_blank()
  )

rotate_x_text_45 = theme(
  axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
)
rotate_x_text_90 = theme(
  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)
)
#### End chart setup ####

create_unique_text <- function(project_title, short_description, long_description) {
  
  project_text <- long_description
  
  if (!grepl(tolower(short_description), tolower(project_text), fixed = TRUE)) {
    project_text <- paste(short_description, project_text)
  }
  
  if (!grepl(tolower(project_title), tolower(project_text), fixed = TRUE)) {
    project_text <- paste(project_title, project_text)
  }
  
  return(str_trim(project_text))
}

# Load data
# Set up parallel processing using available CPU cores
plan(multisession, workers = parallel::detectCores() - 1)

crs = fread("large_input/crs_2014_2023.csv")
crs_text = crs[,c("project_title", "short_description", "long_description"), with=F]
crs_text[, text := future_pmap_chr(.SD, create_unique_text, .progress=T)]
latest_classifications = fread("large_input/crs-2014-2023-housing-labeled-phi4.csv")
latest_classifications$text = trimws(latest_classifications$text)
difs = setdiff(latest_classifications$text, unique(crs_text$text))
latest_classifications$sector_code = NULL

crs$text = crs_text$text
rm(crs_text)
gc()
crs_merge = merge(crs, latest_classifications, by="text")
rm(crs)
gc()

crs_merge$`Sector code` = (crs_merge$sector_code %in% c(16030, 16040))
crs_merge$any = crs_merge$Housing | crs_merge$Homelessness |
  crs_merge$Transitional | crs_merge$Incremental | crs_merge$Social |
  crs_merge$Market | crs_merge$`Sector code`

crs = subset(crs_merge, any==T)

crs = subset(
  crs,
  flow_type_name=="Disbursements" &
    amount_type=="Constant prices"
)

# By disbursement year
crs$`Off continuum` = !crs$Homelessness & 
  !crs$Transitional & 
  !crs$Incremental & 
  !crs$Social &
  !crs$Market
housing_continuum = melt(
  crs,
  id.vars=c("year", "value"),
  measure.vars=c(
    "Homelessness",
    "Transitional",
    "Incremental",
    "Social",
    "Market",
    "Off continuum"
  ),
  value.name = "continuum"
)
housing_continuum_agg = data.table(housing_continuum)[,.(value=sum(value)), by=.(
  year, variable, continuum
)]
housing_continuum_agg = subset(housing_continuum_agg, continuum==T)
crs_agg = crs[,.(total=sum(value)), by=.(year)]
housing_continuum_agg = merge(housing_continuum_agg, crs_agg, by="year")

ggplot(housing_continuum_agg, aes(x=year, y=value, group=variable, fill=variable)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=six_pallette) +
  scale_y_continuous(expand = c(0, 0), n.breaks=6, labels=dollar) +
  scale_x_continuous(n.breaks = 10) +
  expand_limits(y=c(0, max(crs_agg$total*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/virtual_year.png",
  height=5,
  width=8
)
crs_by_year_wide = dcast(housing_continuum_agg, year~variable, value.var="value")
fwrite(crs_by_year_wide, "output/virtual_year.csv")

housing_continuum_agg$percent = housing_continuum_agg$value / housing_continuum_agg$total
percent_year = housing_continuum_agg[,.(percent=sum(percent)), by=.(year)]
ggplot(housing_continuum_agg, aes(x=year, y=percent, group=variable, fill=variable)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=six_pallette) +
  scale_y_continuous(expand = c(0, 0), n.breaks=6, labels=percent) +
  scale_x_continuous(n.breaks = 10) +
  expand_limits(y=c(0, max(percent_year$percent))) +
  custom_style +
  labs(
    y="Housing disbursements\n(% housing dev. finance)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/virtual_year_percent.png",
  height=5,
  width=8
)
percent_year_wide = dcast(housing_continuum_agg, year~variable, value.var="percent")
fwrite(percent_year_wide, "output/virtual_year_percent.csv")

plan(sequential)
