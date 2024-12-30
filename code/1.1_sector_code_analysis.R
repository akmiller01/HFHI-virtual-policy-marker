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
primary_purpose = "#4A207E"

secondary_blue = "#00759F"
secondary_green = "#007B63"
secondary_yellow = "#C69200"
secondary_orange = "#DA5C05"

tertiary_blue = "#93B9DC"
tertiary_grey = "#7C7369"
tertiary_tan = "#D3BE96"

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

# Load data
crs = fread("input/crs_2014_2023_purpose_codes.csv")
crs = subset(
  crs,
  flow_type_name=="Disbursements" &
  amount_type=="Constant prices"
)

# By category
crs_by_category = crs[,.(value=sum(value, na.rm=T)), by=.(year, category_name)]
category_mapping = c(
  "ODA"="ODA",
  "Non-export credit OOF"="OOF",
  "Private Development Finance"="Private",
  "PSI"="Private"
)
crs_by_category$category_name = category_mapping[crs_by_category$category_name]
crs_by_category$category_name = factor(
  crs_by_category$category_name, levels = c("ODA", "OOF", "Private")
)

ggplot(crs_by_category, aes(x=year, y=value, group=category_name, fill=category_name)) +
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
  filename="output/sector_category_year.png",
  height=5,
  width=8
)
crs_by_category_wide = dcast(crs_by_category, year~category_name, value.var="value")
fwrite(crs_by_category_wide, "output/sector_category_year.csv")

# By disbursement year
crs_by_year = crs[,.(value=sum(value, na.rm=T)), by=.(year, sector_name)]

ggplot(crs_by_year, aes(x=year, y=value, group=sector_name, fill=sector_name)) +
  geom_bar(stat="identity") +
  scale_y_continuous(expand = c(0, 0), n.breaks=5, labels=dollar) +
  scale_x_continuous(n.breaks = 10) +
  scale_fill_manual(values=c(master_blue, master_green)) +
  expand_limits(y=c(0, max(crs_by_year$value*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    fill=""
  ) + rotate_x_text_45
ggsave(
  filename="output/sector_year.png",
  height=5,
  width=8
)
crs_by_year_wide = dcast(crs_by_year, year~sector_name, value.var="value")
fwrite(crs_by_year_wide, "output/sector_year.csv")

# By donor
crs_by_donor = crs[,.(value=sum(value, na.rm=T)), by=.(donor_name)]
crs_by_donor = crs_by_donor[order(-crs_by_donor$value),]

donor_short_names = crs_by_donor$donor_name
names(donor_short_names) = crs_by_donor$donor_name
donor_short_names["International Development Association [IDA]"] = 
  "IDA"
donor_short_names["Council of Europe Development Bank"] = 
  "CEB"
donor_short_names["Inter-American Development Bank [IDB]"] = 
  "IDB"
donor_short_names["Asian Development Bank [AsDB]"] = 
  "AsDB"
donor_short_names["International Bank for Reconstruction and Development [IBRD]"] = 
  "IBRD"
donor_short_names["European Bank for Reconstruction and Development [EBRD]"] = 
  "EBRD"
donor_short_names["Asian Infrastructure Investment Bank [AIIB]"] = 
  "AIIB"
fwrite(crs_by_donor, "output/sector_by_donor.csv")
crs_by_donor$donor_name = donor_short_names[crs_by_donor$donor_name]
crs_by_donor$donor_name = factor(
  crs_by_donor$donor_name,
  levels=crs_by_donor$donor_name
)
ggplot(crs_by_donor[1:10], aes(x=donor_name, y=value)) +
  geom_bar(stat="identity",fill=master_blue) +
  scale_y_continuous(expand = c(0, 0), labels=dollar) +
  expand_limits(y=c(0, max(crs_by_donor$value*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    color="",
    title="Top 10 donors to current housing sectors\n(2014-2023)"
  ) +
  rotate_x_text_45
ggsave(
  filename="output/sector_by_donor.png",
  height=5,
  width=8
)

# By donor type
donor_type = fread("input/oecd_crs_donor_type_ref.csv")[,c("donor_code", "donor_type")]
setdiff(unique(crs$donor_code), unique(donor_type$donor_code))
crs_donor_type = merge(crs, donor_type, by="donor_code")
crs_by_donor_type = crs_donor_type[,.(value=sum(value, na.rm=T)), by=.(donor_type)]
crs_by_donor_type = crs_by_donor_type[order(-crs_by_donor_type$value),]

fwrite(crs_by_donor_type, "output/sector_by_donor_type.csv")
crs_by_donor_type$donor_type = factor(
  crs_by_donor_type$donor_type,
  levels=crs_by_donor_type$donor_type
)
ggplot(crs_by_donor_type, aes(x=donor_type, y=value)) +
  geom_bar(stat="identity",fill=master_blue) +
  scale_y_continuous(expand = c(0, 0), labels=dollar) +
  expand_limits(y=c(0, max(crs_by_donor$value*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    color="",
    title="Current housing sector disbursements by donor type\n(2014-2023)"
  ) +
  rotate_x_text_45
ggsave(
  filename="output/sector_by_donor_type.png",
  height=5,
  width=8
)

# By recipient
crs_by_recipient_year = crs[,.(value=sum(value, na.rm=T)), by=.(recipient_name, year)]
crs_by_recipient = crs_by_recipient_year[,.(value=sum(value, na.rm=T)), by=.(recipient_name)]
crs_by_recipient_type = crs_by_recipient

crs_by_recipient = crs_by_recipient[order(-crs_by_recipient$value),]
recipient_short_names = crs_by_recipient$recipient_name
names(recipient_short_names) = crs_by_recipient$recipient_name
# recipient_short_names["X"] =
#   "Y"
fwrite(crs_by_recipient, "output/sector_by_recipient.csv")
crs_by_recipient$recipient_name = recipient_short_names[crs_by_recipient$recipient_name]
crs_by_recipient$recipient_name = factor(
  crs_by_recipient$recipient_name,
  levels=crs_by_recipient$recipient_name
)
ggplot(crs_by_recipient[1:10], aes(x=recipient_name, y=value)) +
  geom_bar(stat="identity",fill=master_blue) +
  scale_y_continuous(expand = c(0, 0), labels=dollar) +
  expand_limits(y=c(0, max(crs_by_recipient$value*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    color="",
    title="Top 10 recipients of current housing sectors\n(2014-2023)"
  ) +
  rotate_x_text_45
ggsave(
  filename="output/sector_by_recipient.png",
  height=5,
  width=8
)

# By recipient income group
income_groups = unique(crs[,c("recipient_name", "income_group_code")])
crs_by_recipient_type = merge(crs_by_recipient_type, income_groups, by="recipient_name")
crs_by_recipient_type = crs_by_recipient_type[,.(
  value=sum(value, na.rm=T)
), by=.(income_group_code)]

crs_by_recipient_type = crs_by_recipient_type[order(-crs_by_recipient_type$value),]
fwrite(crs_by_recipient_type, "output/sector_by_income.csv")
crs_by_recipient_type$income_group_code = factor(
  crs_by_recipient_type$income_group_code,
  levels=crs_by_recipient_type$income_group_code
)
ggplot(crs_by_recipient_type, aes(x=income_group_code, y=value)) +
  geom_bar(stat="identity",fill=master_blue) +
  scale_y_continuous(expand = c(0, 0), labels=dollar) +
  expand_limits(y=c(0, max(crs_by_recipient_type$value*1.1))) +
  custom_style +
  labs(
    y="Housing disbursements\n(constant 2022 US$ millions)",
    x="",
    color="",
    title="Current housing sector disbursements by income group\n(2014-2023)"
  ) +
  rotate_x_text_45
ggsave(
  filename="output/sector_by_income.png",
  height=5,
  width=8
)
