#### Setup ####
list.of.packages <- c("data.table", "rstudioapi", "scales","ggplot2","scales","Hmisc","openxlsx", "countrycode")
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

crs = fread("large_output/crs_2024_update_preprocessed_original_labeled_wb.csv")
original_count = nrow(crs)

crs$`Sector code` = (crs$PurposeCode %in% c(16030, 16040))
crs$any = crs$`Transitional and Temporary Housing` | crs$`Resilience and Reconstruction` |
  crs$`Incremental and Improved Housing` | crs$`Social Housing` |
  crs$`Market Enabling` | crs$`Housing Supply` | crs$`Sector code`
crs$any_sans_sector = crs$`Transitional and Temporary Housing` | crs$`Resilience and Reconstruction` |
  crs$`Incremental and Improved Housing` | crs$`Social Housing` |
  crs$`Market Enabling` | crs$`Housing Supply`

potential_false_negative = subset(crs, `Sector code` & !any_sans_sector)
crs = subset(crs, any==T)
crs$Year = as.numeric(crs$Year)
selected_count = nrow(crs)
percent_format(0.01)(selected_count/original_count)
sum(crs$USD_Disbursement_Defl, na.rm=T)

# By disbursement year
crs$`Housing Supply` = !crs$`Transitional and Temporary Housing` &
  !crs$`Resilience and Reconstruction` &
  !crs$`Incremental and Improved Housing` &
  !crs$`Social Housing` &
  !crs$`Market Enabling`
fwrite(crs, "large_output/crs_2024_update_housing_wb.csv")
housing_continuum = melt(
  crs,
  id.vars=c("Year", "USD_Disbursement_Defl"),
  measure.vars=c(
    "Transitional and Temporary Housing",
    "Resilience and Reconstruction",
    "Incremental and Improved Housing",
    "Social Housing",
    "Market Enabling",
    "Housing Supply"
  ),
  value.name = "continuum"
)
housing_continuum_agg = data.table(housing_continuum)[,.(USD_Disbursement_Defl=sum(USD_Disbursement_Defl)), by=.(
  Year, variable, continuum
)]
housing_continuum_agg = subset(housing_continuum_agg, continuum==T)
crs_agg = crs[,.(total=sum(USD_Disbursement_Defl)), by=.(Year)]
housing_continuum_agg = merge(housing_continuum_agg, crs_agg, by="Year")
ggplot(housing_continuum_agg, aes(x=Year, y=USD_Disbursement_Defl, group=variable, fill=variable)) +
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
  filename="output/wb_virtual_year.png",
  height=5,
  width=8
)
crs_by_year_wide = dcast(housing_continuum_agg, Year~variable, value.var="USD_Disbursement_Defl")
fwrite(crs_by_year_wide, "output/wb_virtual_year.csv")

housing_continuum_agg$percent = housing_continuum_agg$USD_Disbursement_Defl / housing_continuum_agg$total
percent_year = housing_continuum_agg[,.(percent=sum(percent)), by=.(Year)]
ggplot(housing_continuum_agg, aes(x=Year, y=percent, group=variable, fill=variable)) +
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
  filename="output/wb_virtual_year_percent.png",
  height=5,
  width=8
)
percent_year_wide = dcast(housing_continuum_agg, Year~variable, value.var="percent")
fwrite(percent_year_wide, "output/wb_virtual_year_percent.csv")

# Category
crs_by_category = crs[,.(value=sum(USD_Disbursement_Defl, na.rm=T)), by=.(Year, Category)]
category_codes = c(
  "10"="ODA",
  "21"="Non-export credit OOF",
  "22"="Officially supported export credits",
  "30"="Private Development Finance",
  "36"="Private Foreign Direct Investment",
  "37"="Other Private flows at market terms",
  "40"="Non flow",
  "50"="Other flows",
  "60"="PSI"
)
crs_by_category$category_name = category_codes[as.character(crs_by_category$Category)]
category_mapping = c(
  "ODA"="ODA",
  "Non-export credit OOF"="OOF",
  "Private Development Finance"="Private",
  "PSI"="Private"
)
crs_by_category$category_name = category_mapping[crs_by_category$category_name]
crs_by_category = crs_by_category[,.(value=sum(value)),by=.(Year, category_name)]
crs_by_category$category_name = factor(
  crs_by_category$category_name, levels = c("ODA", "OOF", "Private")
)
ggplot(crs_by_category, aes(x=Year, y=value, group=category_name, fill=category_name)) +
  geom_bar(stat="identity") +
  scale_y_continuous(expand = c(0, 0), n.breaks=5, labels=dollar) +
  scale_x_continuous(n.breaks = 10) +
  scale_fill_manual(values=c(master_blue, secondary_blue, master_green, secondary_green)) +
  expand_limits(y=c(0, max(crs_by_category$value*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/wb_virtual_category_Year.png",
  height=5,
  width=8
)
crs_by_category_wide = dcast(crs_by_category, Year~category_name, value.var="value")
fwrite(crs_by_category_wide, "output/wb_virtual_category_Year.csv")

# Urban/rural
housing_urban_rural = melt(
  crs,
  id.vars=c("Year", "USD_Disbursement_Defl"),
  measure.vars=c(
    "Urban", "Rural"
  ),
  value.name = "bool"
)
ur_agg = data.table(housing_urban_rural)[,.(USD_Disbursement_Defl=sum(USD_Disbursement_Defl)), by=.(
  Year, variable, bool
)]
ur_agg = subset(ur_agg, bool==T)
ur_agg_agg = ur_agg[,.(total=sum(USD_Disbursement_Defl)), by=.(Year)]
ur_agg = merge(ur_agg, ur_agg_agg, by="Year")

ur_agg$percent = ur_agg$USD_Disbursement_Defl / ur_agg$total
percent_year = ur_agg[,.(percent=sum(percent)), by=.(Year)]
ggplot(ur_agg, aes(x=Year, y=percent, group=variable, fill=variable)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=c(master_blue, master_green)) +
  scale_y_continuous(expand = c(0, 0), n.breaks=6, labels=percent) +
  scale_x_continuous(n.breaks = 10) +
  expand_limits(y=c(0, max(percent_year$percent))) +
  custom_style +
  labs(
    y="Housing disbursements\n(% located housing dev. finance)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/wb_virtual_urban_rural_year_percent.png",
  height=5,
  width=8
)
urban_rural_year_wide = dcast(ur_agg, Year~variable, value.var="percent")
fwrite(urban_rural_year_wide, "output/wb_virtualurban_rural_year_percent.csv")