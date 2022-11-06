import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter


data = pd.read_csv("data_2.csv")

High = data.query("Type == 3")
Medium = data.query("Type == 2")
Low = data.query("Type == 1")

########################### Kaplan-Meier se olo to dataset ##################################
kmf = KaplanMeierFitter()
kmf.fit(durations = data["Tool wear [min]"], event_observed = data["Machine failure"])
kmf.plot()

plt.title('Kaplan-Meier Estimate')
plt.ylabel("Probability of machine still working")
plt.show()
############################################################################################


#################### Kaplan-Meier cumulative density se olo to dataset #####################
kmf = KaplanMeierFitter()
kmf.fit(durations = data["Tool wear [min]"], event_observed = data["Machine failure"])
kmf.plot_cumulative_density()
print(kmf.cumulative_density_)

plt.title('Cumulative Density')
plt.ylabel("Probability of machine still working")
plt.show()
############################################################################################

########################### Kaplan-Meier me bash to type  ##################################
kmf_h = KaplanMeierFitter()
kmf_m = KaplanMeierFitter()
kmf_l = KaplanMeierFitter()

kmf_h.fit(durations = High["Tool wear [min]"],event_observed = High["Machine failure"],label="High")
kmf_m.fit(durations = Medium["Tool wear [min]"],event_observed = Medium["Machine failure"],label="Medium")
kmf_l.fit(durations = Low["Tool wear [min]"],event_observed = Low["Machine failure"],label="Low")

kmf_h.plot()
kmf_m.plot()
kmf_l.plot()

plt.title('Kaplan-Meier Estimate')
plt.ylabel("Probability of machine still working")
plt.show()
############################################################################################

##################### Kaplan-Meier cumulative density me bash to type ######################
kmf_h = KaplanMeierFitter()
kmf_m = KaplanMeierFitter()
kmf_l = KaplanMeierFitter()

kmf_h.fit(durations = High["Tool wear [min]"],event_observed = High["Machine failure"],label="High")
kmf_m.fit(durations = Medium["Tool wear [min]"],event_observed = Medium["Machine failure"],label="Medium")
kmf_l.fit(durations = Low["Tool wear [min]"],event_observed = Low["Machine failure"],label="Low")

kmf_h.plot_cumulative_density()
kmf_m.plot_cumulative_density()
kmf_l.plot_cumulative_density()

plt.title('Cumulative Density')
plt.ylabel("Probability of machine still working")
plt.show()
############################################################################################

##################### Nelson-Aalencumulative density se olo to dataset ######################
naf = NelsonAalenFitter()
naf.fit(durations = data["Tool wear [min]"], event_observed = data["Machine failure"])
naf.plot_cumulative_hazard()

plt.title("Nelson-Aalen Cumulative Hazard")
plt.ylabel("Probability of machine still working")
plt.show()
############################################################################################

##################### Nelson-Aalencumulative density me bash to type #######################
naf_h = NelsonAalenFitter()
naf_m = NelsonAalenFitter()
naf_l = NelsonAalenFitter()

naf_h.fit(durations = High["Tool wear [min]"],event_observed = High["Machine failure"],label="High")
naf_m.fit(durations = Medium["Tool wear [min]"],event_observed = Medium["Machine failure"],label="Medium")
naf_l.fit(durations = Low["Tool wear [min]"],event_observed = Low["Machine failure"],label="Low")

naf_h.plot_cumulative_hazard()
naf_m.plot_cumulative_hazard()
naf_l.plot_cumulative_hazard()

plt.title("Nelson-Aalen Cumulative Hazard")
plt.ylabel("Probability of machine still working")
plt.show()
############################################################################################

################################ Log-rank test sta types ####################################
T=High["Tool wear [min]"]
E=High["Machine failure"]
T1=Medium["Tool wear [min]"]
E1=Medium["Machine failure"]
T2=Low["Tool wear [min]"]
E2=Low["Machine failure"]

results=logrank_test(T,T1,event_observed_A=E, event_observed_B=E1)
results.print_summary()
###########################################################################################

##################################### Cox regression ######################################
data = data[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Machine failure']]

cph = CoxPHFitter()
cph.fit(data,"Tool wear [min]",event_col="Machine failure")
cph.print_summary()

cph.plot()
plt.title('Cox')
plt.show()
###########################################################################################

############################################ CTE ##########################################
kmf = KaplanMeierFitter()
kmf.fit(durations = data["Tool wear [min]"], event_observed = data["Machine failure"])

CTE = kmf.conditional_time_to_event_
plt.plot(CTE)

plt.title('CTE')
plt.show()
###########################################################################################