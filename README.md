# H1N1-Flu-Vaccine
Predicting the likelihood of an individual receiving the H1N1 and seasonal flu vaccines.

## Problem Statement
Since the rise of the COVID-19, public health experts across the world have pointed to the development of 
a vaccine as a key factor in stemming the spread of the disease. Vaccines provide acquired immunity to 
infectious diseases for individuals, and enough immunizations in a community can further reduce the 
spread of diseases through "herd immunity."<sup>1</sup> While the promise of a vaccine provides hope to infectious 
disease experts on defeating the virus, individuals need to be willing to getting the vaccine. An NPR poll 
recently found that only half of Americans say they will get a COVID-19 vaccine once available. Respondents 
cited concerns about the side effects and fears about contracting the virus as reasons for not wanting to 
receive the vaccine.<sup>2</sup>

To explore public health response to a different but recent major respiratory disease pandemic, 
DataDriven.org is hosting a competition around the likelihood of individuals to receive the H1N1 and 
seasonal flu vaccines. H1N1 is the subtype of Influenza A virus and well known outbreaks of H1N1 strains 
occurred during the 2009 swine flue pandemic as well as the 1918 “Spanish” Flu Pandemic. A vaccine for 
the H1N1 flu virus became publicly available in October 2009. The competition aims to explore how we 
can predict if an individual will get a COVID-19 vaccine by looking at data on the H1N1 and seasonal 
vaccines from data from 2009-2010. A better understanding of how these characteristics are associated 
with personal vaccination patterns can provide guidance for future public health efforts. The goal is to use 
these characteristics to predict how likely an individual is to receive their H1N1 and seasonal flu vaccines.

## Data Source
In late 2009 and early 2010, the United States National Center for Health Statistics conducted the National 
2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and 
seasonal flu vaccines, in addition to questions about the respondents. These additional questions covered 
their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness,
and behaviors towards mitigating transmission.
The dataset contains the survey responses from 26,706 respondents and includes 35 features, which are 
detailed in Appendix A – Dataset Features. Of the 35 features 13 are binary, 12 are categorical, 8 are ordinal, 
and 2 are numerical. The dataset includes two response variables which are binary and represent whether or not 
respondents reported having received a H1N1 vaccine or a seasonal flu vaccine. 

Figure 1 and Figure 2 display
the proportion of respondents who received the H1N1 and seasonal flu vaccines respectively. Only 21% of respondents 
received the H1N1 vaccine, while 47% received the seasonal flu vaccine.
