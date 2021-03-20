'''
Tony Shu
Electricity Market
'''

import logging
#Comment line below to display optimization outputs ie. Runtime, Variable count, etc.
'''-------------------------------------'''
logging.basicConfig(level=logging.ERROR)
'''-------------------------------------'''

import pypsa
import pypsa.opf
from pypsa.opt import (l_constraint, l_objective, LExpression, LConstraint,
				  patch_optsolver_free_model_before_solving,
				  patch_optsolver_record_memusage_before_solving,
				  empty_network, free_pyomo_initializers)
				  
from pypsa.opt import (l_constraint, l_objective, LExpression, LConstraint,
					patch_optsolver_free_model_before_solving,
					patch_optsolver_record_memusage_before_solving,
					empty_network, free_pyomo_initializers)
					
from pypsa.descriptors import (get_switchable_as_dense, get_switchable_as_iter,allocate_series_dataframes, zsum, Dict)

pypsa.pf.logger.setLevel(logging.ERROR) #Disable extra output from optimization

import numpy as np
import pandas as pd #Only v.23 works as .24 removes sort in append
import os
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import random
import re
import time
import copy
import sys
import datetime


from six.moves import map, zip, range, reduce
from six import itervalues, iteritems
import six


from pyomo.environ import (ConcreteModel, Var, Objective,
                           NonNegativeReals, Constraint, Reals,
                           Suffix, Expression, Binary, SolverFactory)


'Plotting Libraries'
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid
from bokeh.models.glyphs import Step
from bokeh.io import curdoc, show

'Math Libraries'
import statsmodels.tsa.stattools as ts 
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import normalize

'''
GLOBAL VARIABLES
'''
OPTIMIZATION_SOLVER = 'cbc'


CARTOGRAPHY_DATA = 'cartopy_data_dir\\'
cartopy.config['data_dir'] = CARTOGRAPHY_DATA


#INDIA_DATA_DIR = 'india_data\\'
INDIA_DATA_DIR = 'india_data_cases\\'
INDIA_NETWORK_SCENARIO = 'L,H,8,1275'
SIM_OUTPUT_DIR = 'sim_output\\'
TABLEAU_OUTPUT_DIR = 'tableau_output\\'
DEMAND_RESPONSE_BIDS = 'aggregator_data\\Generated_Data\\med-night-bids.csv'
DEFAULT_NO_CHARGING = False
BIDS_FOLDER = 'aggregator_data\\Generated_Data\\'
SCENARIO_NAME = 'med-night'
ALL_SCENARIOS_OUTPUT_FOLDER = 'output_scenarios\\'
V2G_BIDS = 'aggregator_data\\V2G_bids\\v2g_bids_med_lowered.csv'

STARTING_TIME_BLOCK = 0

#DEMAND_RESPONSE_BIDS = 'india_data_cases\\Generated_EV_Data\\Generated_EV_bids_data.csv'


#take a copy of the components pandas.DataFrame
#create a pandas.DataFrame with the properties of the new component attributes.
#the format should be the same as pypsa/component_attrs/*.csv
override_components = pypsa.components.components.copy()
override_component_attrs = Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})

MIN_SYS_RESERVE_ENERGY_MW = 250.0
CVOLL = 9000.0
CRESERVE = 1000.0
CDRCURTAIL = 600.0
CURRENT_TIME_INDEX = 0
CURRENT_HOUR_INDEX = 0 #Current hour of the day



GENERATOR_CLUSTER_SIZE = 10 #Capture 25%, 50%, 75%, or 100% of fleet on or off
GENERATOR_OUTAGE = {}
GENERATOR_CLUSTER_OUTAGE = []
DAILY_FIXED_DR = None
#MIN_SYS_RESERVE_ENERGY_MW = 0.0


'''
-----------------
'''


'''
Utility Functions
'''
def MW_to_GW(mw_pow):
	return mw_pow/1000.0

def GW_to_MW(gw_pow):
	return gw_pow*1000.0
	
def write_output(filename,string, output_dir = None):

	if output_dir is None:
		with open(SIM_OUTPUT_DIR + filename,'w') as outfile:
			outfile.write(string)
	else:
		with open(output_dir + filename, 'w') as outfile:
			outfile.write(string)
		
def write_tableau_output(output_dir,rtm_networks,dam_uc_networks,dam_ed_networks, generator_data, cluster_generator_data, 
						dam_dr_data, outage_data, dam_vre_available, dam_curtailment, dam_unmet_demand, dam_unmet_reserve, 
						rtm_unmet_demand, rtm_unmet_reserve, rtm_vre_available, rtm_curtailment, rtm_dr_curtailment, dam_reserve_data,
						time_block, scenario_name, write_type):
	'''Write the outputs for Tableau Visualization'''
	global CURRENT_TIME_INDEX
	
	
	'Demand Response Data'
	
	if write_type == 'w':
		dr_data_str = 'Zone,Attribute,Value,Timeblock,Scenario\n'
		rtm_dr_data_str = 'Zone,Attribute,Value,Timeblock,Scenario\n'
	else:
		dr_data_str = ''
		rtm_dr_data_str = ''
	
	#DAM DR
	day = 0
	for dr_data in dam_dr_data:
		for (zone, snapshot) in dr_data.keys():
			dr_data_str = dr_data_str + str(zone) + ',t' + str(snapshot + 1 + (24 * day)) + ',' + str(dr_data[(zone,snapshot)]) + ',t' + str(time_block) + ',' + scenario_name + '\n'
		day = day + 1
	
	#RTM DR
	hour = 0
	for dr_data in rtm_dr_curtailment:
		if dr_data is not None:
			for (zone, snapshot) in dr_data.keys():
				rtm_dr_data_str = rtm_dr_data_str + str(zone) + ',t' + str(snapshot + 1 + (4 * hour)) + ',' + str(dr_data[(zone,snapshot)]) + ',t' + str(time_block) + ',' + scenario_name + '\n'
		hour = hour + 1
		
	with open(output_dir + 'tab_dam_demand_response_load.csv',write_type) as outfile:
		outfile.write(dr_data_str)
	with open(output_dir + 'tab_rtm_demand_response_curtailment.csv',write_type) as outfile:
		outfile.write(rtm_dr_data_str)
	'''---'''
	
	'DAM Data'
	if write_type == 'w':
		dam_gen_p_str = 'Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
		dam_storage_p_str = 'Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
		dam_storage_soc_str = 'Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
		dam_lmp_str = 'Zone,Attribute,Value,Timeblock,Scenario\n'
		dam_load_str = 'Zone,Attribute,Value,Timeblock,Scenario\n'
		dam_reserve_str = 'Resource,Attribute,Value,Timeblock,Scenario\n'
	else:
		dam_gen_p_str = ''
		dam_storage_p_str = ''
		dam_storage_soc_str = ''
		dam_lmp_str = ''
		dam_load_str = ''
		dam_reserve_str = ''
		
	#UC
	total_t = 0
	for scuc_network in dam_uc_networks:
	
		cur_day = int(total_t / 24.0)
		
		dam_gens = scuc_network.generators_t
		for (name,gen_power) in dam_gens.p.iteritems():
			t = total_t + 1
			
			gen_id = int(re.split('([\d]+)',name)[1])
			try:
				resource_name = str(generator_data[gen_id]['Resource'])
			except:
				resource_name = str(cluster_generator_data[gen_id]['Resource'])
			
			'VRE Generation Output is Different'
			if resource_name == 'Solar' or resource_name == 'Wind':
				
				#Printing total VRE available
				for avail_pow in dam_vre_available[cur_day][name]:
					dam_gen_p_str = dam_gen_p_str + ( resource_name + ',' + str(generator_data[gen_id]['zone'])  
												  + ',t' + str(t) + ',' + str(avail_pow) + ',t' + str(time_block) + ',' + scenario_name + '\n')
					t = t + 1
					
				#Printing total curtail from each VRE Gen
				t = total_t + 1
				for curtail_pow in dam_curtailment[cur_day][name]:
					dam_gen_p_str = dam_gen_p_str + ('Curtailment' + ',' + str(generator_data[gen_id]['zone'])  
												  + ',t' + str(t) + ',' + str(np.abs(curtail_pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n')
					t = t + 1
			else:
				for pow in gen_power:
					#Clustering causes new generator IDs
					try:
						dam_gen_p_str = dam_gen_p_str + ( resource_name + ',' + str(generator_data[gen_id]['zone']) 
													  + ',t' + str(t) + ',' + str(np.abs(pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n')
					except:
						dam_gen_p_str = dam_gen_p_str + (resource_name + ',' + str(cluster_generator_data[gen_id]['zone']) 
													 + ',t' + str(t) + ',' + str(np.abs(pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n')			
					t = t + 1
			
		dam_storage_units = scuc_network.storage_units_t
		for (name, stor_power) in dam_storage_units.p.iteritems():
			t = total_t + 1
			for pow in stor_power:
				gen_id = int(re.split('([\d]+)',name)[1])
				dam_storage_p_str = dam_storage_p_str + (str(generator_data[gen_id]['Resource']) + ',' + str(generator_data[gen_id]['zone']) 
													 + ',t' + str(t) + ',' + str(pow) + ',t' + str(time_block) + ',' + scenario_name + '\n')
				t = t + 1
				
		for (name, stor_soc) in dam_storage_units.state_of_charge.iteritems():
			t = total_t + 1
			for soc in stor_soc:
				gen_id = int(re.split('([\d]+)',name)[1])
				dam_storage_soc_str = dam_storage_soc_str + (str(generator_data[gen_id]['Resource']) + ',' + str(generator_data[gen_id]['zone']) 
													 + ',t' + str(t) + ',' + str(np.abs(soc)) + ',t' + str(time_block) + ',' + scenario_name + '\n')										 
				t = t + 1
		
		dam_loads = scuc_network.loads_t
		zone = 0
		for (name,zone_loads) in dam_loads.p.iteritems():
			t = total_t + 1
			zone = zone + 1
			for load in zone_loads:
				dam_load_str = dam_load_str + str(zone) + ',t' + str(t) + ',' + str(load) + ',t' + str(time_block) + ',' + scenario_name + '\n'
				t = t + 1
		total_t = total_t + 24
	#ED
	total_t = 0
	for scuc_price_network in dam_ed_networks:
		dam_buses = scuc_price_network.buses_t
		for (name,bus_price) in dam_buses.marginal_price.iteritems():
			t = total_t + 1
			for price in bus_price:
				dam_lmp_str = dam_lmp_str + str(name) + ',t' + str(t) + ',' + str(price) + ',t' + str(time_block) + ',' + scenario_name + '\n'
				t = t + 1
		total_t = total_t + 24
	
	#Unmet Variables
	day = 0
	for unmet_demand_data in dam_unmet_demand:
		for (zone, snapshot) in unmet_demand_data.keys():
			dam_gen_p_str = dam_gen_p_str + ( 'Unmet Demand' + ',' + str(zone) 
										  + ',t' + str(snapshot + 1 + (24 * day)) + ',' + str(unmet_demand_data[(zone,snapshot)]) 
										  + ',t' + str(time_block) + ',' + scenario_name + '\n')
		day = day + 1
	
	#DAM Reserves
	day = 0
	for reserve_data in dam_reserve_data:
		for (unit_id, snapshot) in reserve_data.keys():
			gen_id = int(re.split('([\d]+)',unit_id)[1])
			try:
				resource_name = str(generator_data[gen_id]['Resource'])
			except:
				resource_name = str(cluster_generator_data[gen_id]['Resource'])
			dam_reserve_str = dam_reserve_str + (resource_name + ',t' + str(snapshot + 1 + (24 * day)) + ',' + str(reserve_data[(unit_id,snapshot)])
												 + ',t' + str(time_block) + ',' + scenario_name + '\n')
		day = day + 1
		
	with open(output_dir + 'tab_dam_generator_power.csv',write_type) as outfile:
		outfile.write(dam_gen_p_str)
	with open(output_dir + 'tab_dam_prices.csv',write_type) as outfile:
		outfile.write(dam_lmp_str)
	with open(output_dir + 'tab_dam_storage_unit_power.csv',write_type) as outfile:
		outfile.write(dam_storage_p_str)
	with open(output_dir + 'tab_dam_storage_unit_soc.csv',write_type) as outfile:
		outfile.write(dam_storage_soc_str)
	with open(output_dir + 'tab_dam_loads.csv',write_type) as outfile:
		outfile.write(dam_load_str)
	with open(output_dir + 'tab_dam_reserves.csv',write_type) as outfile:
		outfile.write(dam_reserve_str)
		
	'''---'''
	
	'RTM Data'
	if write_type == 'w':
		rtm_gen_p_str = 'Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
		rtm_price_str = 'Zone,Attribute,Value,Timeblock,Scenario\n'
		rtm_load_str = 'Zone,Attribute,Value,Timeblock,Scenario\n'
		rtm_storage_p_str = 'Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
		rtm_storage_soc_str = 'Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
	else:
		rtm_gen_p_str = ''
		rtm_price_str = ''
		rtm_load_str = ''
		rtm_storage_p_str = ''
		rtm_storage_soc_str = ''
		
	total_t = 0
	for rtm_network in rtm_networks:
	
		cur_hour = int(total_t / 4.0)
		
		#Power
		rtm_gens = rtm_network.generators_t
		for (name, gen_power) in rtm_gens.p.iteritems():
			t = total_t + 1
			for pow in gen_power:
				gen_id = int(re.split('([\d]+)',name)[1])
				
				'V2G resources should have V2G in the name'
				if 'V2G' in name:
					rtm_gen_p_str = rtm_gen_p_str + re.split('V2G ([a-zA-Z\s]+) ([\d]+)',name)[1] + ',1,t' + str(t) + ',' + str(np.abs(pow)) + ',t' + str(time_block)  + ',' + scenario_name + '\n'
				else:
					try:
						resource_name = str(generator_data[gen_id]['Resource']) #.replace('Old','').replace('New','')
						
						'Printing the curtailment and available VRE of Solar and Wind'
						#if False: #resource_name == 'Solar' or resource_name == 'Wind':
						if resource_name == 'Solar' or resource_name == 'Wind':
							#DO NOTHING WILL PRINT VRE IN DIFFERENT SET
							'''
							#Printing total VRE available
							for avail_pow in rtm_vre_available[cur_hour][name]:
								rtm_gen_p_str = rtm_gen_p_str + ( resource_name + ',' + str(generator_data[gen_id]['zone'])  
															  + ',t' + str(t) + ',' + str(avail_pow) + ',t' + str(time_block) + ',' + scenario_name + '\n')
								t = t + 1
					
							#Printing total curtail from each VRE Gen
							t = total_t + 1
							for curtail_pow in rtm_curtailment[cur_hour][name]:
								rtm_gen_p_str = rtm_gen_p_str + ('Curtailment' + ',' + str(generator_data[gen_id]['zone'])  
															  + ',t' + str(t) + ',' + str(np.abs(curtail_pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n')
								t = t + 1
							'''
						else:
							rtm_gen_p_str = rtm_gen_p_str + resource_name + ',' + str(generator_data[gen_id]['zone']) + ',t' + str(t) + ',' + str(np.abs(pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n'
					except:
						try:
							# Clustered Generators
							resource_name = str(cluster_generator_data[gen_id]['Resource']) #.replace('Old','').replace('New','')
							rtm_gen_p_str = rtm_gen_p_str + resource_name + ',' + str(cluster_generator_data[gen_id]['zone']) + ',t' + str(t) + ',' + str(np.abs(pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n'
						except:
							# V2G Outputs as their IDS should not be in the generator dictionary
							rtm_gen_p_str = rtm_gen_p_str + 'V2G,1,t' + str(t) + ',' + str(np.abs(pow)) + ',t' + str(time_block)  + ',' + scenario_name + '\n'
				t = t + 1
		
		
		
		#VRE Power
		rtm_gens = rtm_network.generators_t
		for (name, gen_power) in rtm_gens.p.iteritems():
			t = total_t + 1
			#for pow in gen_power:
			gen_id = int(re.split('([\d]+)',name)[1])
			try:
				resource_name = str(generator_data[gen_id]['Resource']) #.replace('Old','').replace('New','')
				
				'Printing the curtailment and available VRE of Solar and Wind'
				if resource_name == 'Solar' or resource_name == 'Wind':
					#Printing total VRE available
					for avail_pow in rtm_vre_available[cur_hour][name]:
						#print(avail_pow)
							
						rtm_gen_p_str = rtm_gen_p_str + ( resource_name + ',' + str(generator_data[gen_id]['zone'])  
													  + ',t' + str(t) + ',' + str(avail_pow) + ',t' + str(time_block) + ',' + scenario_name + '\n')
						t = t + 1
					
					#Printing total curtail from each VRE Gen
					t = total_t + 1
					for curtail_pow in rtm_curtailment[cur_hour][name]:
						rtm_gen_p_str = rtm_gen_p_str + ('Curtailment' + ',' + str(generator_data[gen_id]['zone'])  
													  + ',t' + str(t) + ',' + str(np.abs(curtail_pow)) + ',t' + str(time_block) + ',' + scenario_name + '\n')
						t = t + 1
			except:
				'other resources'
		
		#Storage
		rtm_storage_units = rtm_network.storage_units_t
		for (name, stor_power) in rtm_storage_units.p.iteritems():
			t = total_t + 1
			for pow in stor_power:
				gen_id = int(re.split('([\d]+)',name)[1])
				rtm_storage_p_str = rtm_storage_p_str + (str(generator_data[gen_id]['Resource']) + ',' + str(generator_data[gen_id]['zone']) 
													 + ',t' + str(t) + ',' + str(pow) + ',t' + str(time_block) + ',' + scenario_name + '\n')
				t = t + 1
		for (name, stor_soc) in rtm_storage_units.state_of_charge.iteritems():
			t = total_t + 1
			for soc in stor_soc:
				gen_id = int(re.split('([\d]+)',name)[1])
				rtm_storage_soc_str = rtm_storage_soc_str + (str(generator_data[gen_id]['Resource']) + ',' + str(generator_data[gen_id]['zone']) 
													 + ',t' + str(t) + ',' + str(np.abs(soc)) + ',t' + str(time_block) + ',' + scenario_name + '\n')										 
				t = t + 1
		#Prices
		rtm_buses= rtm_network.buses_t
		for (name, bus_price) in rtm_buses.marginal_price.iteritems():
			t = total_t + 1
			for price in bus_price:
				rtm_price_str = rtm_price_str + str(name) + ',t' + str(t) + ',' + str(price) + ',t' + str(time_block) + ',' + scenario_name + '\n'
				t = t + 1
		
		rtm_loads = rtm_network.loads_t
		zone = 0
		for (name,zone_loads) in rtm_loads.p.iteritems():
			t = total_t + 1
			zone = zone + 1
			for load in zone_loads:
				rtm_load_str = rtm_load_str + str(zone) + ',t' + str(t) + ',' + str(load) + ',t' + str(time_block) + ',' + scenario_name + '\n'
				t = t + 1
		total_t = total_t + 4
	
	#Unmet Variables
	day = 0
	hour = 0
	for unmet_demand_data in rtm_unmet_demand:
		for (zone, snapshot) in unmet_demand_data.keys():
			rtm_gen_p_str = rtm_gen_p_str + ('Unmet Demand' + ',' + str(zone) 
										  + ',t' + str(snapshot + 1 + (4 * hour)) + ',' + str(unmet_demand_data[(zone,snapshot)]) 
										  + ',t' + str(time_block) + ',' + scenario_name + '\n')
		hour = hour + 1
	
	with open(output_dir + 'tab_rtm_generator_power.csv',write_type) as outfile:
		outfile.write(rtm_gen_p_str)
	with open(output_dir + 'tab_rtm_prices.csv',write_type) as outfile:
		outfile.write(rtm_price_str)
	with open(output_dir + 'tab_rtm_loads.csv',write_type) as outfile:
		outfile.write(rtm_load_str)
	with open(output_dir + 'tab_rtm_storage_unit_power.csv',write_type) as outfile:
		outfile.write(rtm_storage_p_str)
	with open(output_dir + 'tab_rtm_storage_unit_soc.csv',write_type) as outfile:
		outfile.write(rtm_storage_soc_str)
	'''---'''
	
	'Outage Data'
	if write_type == 'w':
		outage_generator_str = 'Generator_ID,Resource,Zone,Attribute,Value,Timeblock,Scenario\n'
	else:
		outage_generator_str = ''
		
	total_t = 0
	for day in range(len(outage_data)):
		for (gen_id,current_snapshot_outage,outage_left) in outage_data[day]:
			t = 24 * day
			for out_ctr in range(outage_left):
				outage_generator_str = (outage_generator_str + str(gen_id) + ',' + str(cluster_generator_data[gen_id]['Resource']) + 
										',' + str(cluster_generator_data[gen_id]['zone']) + ',t' + str(t + out_ctr) + 
										',' + str(np.abs(current_snapshot_outage)) + ',t' + str(time_block) + ',' + scenario_name + '\n')
				
	with open(output_dir + 'tab_gen_outage.csv',write_type) as outfile:
		outfile.write(outage_generator_str)

'''
-----------------
'''

'''
Data Import Functions
'''

def read_india_generators():
	'R_ID,zone,voltage_level,Resource,RENEW,THERM,DISP,NDISP,STOR,DR,HEAT,NACC,HYDRO,VRE,'
	'Commit,Min_Share,Max_Share,Existing_Cap_MW,New_Build,Cap_size,Max_Cap_MW,Min_Cap_MW,'
	'Min_Share_percent,Max_Share_percent,Inv_cost_per_MWyr,Inv_cost_per_Mwhyr,Fixed_OM_cost_per_MWyr,'
	'Var_OM_cost_per_MWh,Externality_cost_MWh,Start_cost,Start_fuel_MMBTU_per_start,Heat_rate_MMBTU_per_MWh,'
	'Fuel,Min_power,Self_disch,Eff_up,Eff_down,Ratio_power_to_energy,Max_DSM_delay,Ramp_Up_percentage,'
	'Ramp_Dn_percentage,Up_time,Down_time,NACC_Eff,NACC_Peak_to_Base,Reg_Up,Reg_Dn,Rsv_Up,Rsv_Dn,Reg_Cost,'
	'Rsv_Cost,Fixed_OM_cost_per_MWhyr,Var_OM_cost_per_MWh_in,Hydro_level'
	
	#df = pd.read_csv(INDIA_DATA_DIR + 'Generators_data_New_Scenarios.csv')
	print('Reading Generator Data: ' + INDIA_DATA_DIR + 'Generators_data.csv')
	df = pd.read_csv(INDIA_DATA_DIR + 'Generators_data.csv')
	generators = {}
	cluster_generators = {}
	#cluster_resources = ['CCGT Old', 'CCGT New', 'Coal Old', 'Coal new', 'Nuclear Old', 'Nuclear New', 'Biomass']
	cluster_resources = ['CCGT Old', 'CCGT New' , 'Coal Old', 'Coal new', 'Nuclear Old', 'Nuclear New', 'Biomass']
	#cluster_resources = ['CCGT New',  'Coal New']
	fuels = read_india_fuels()
	
	
	for (index, gen_data) in df.iterrows():
	
			'Convert KW to MW'
			generators[gen_data['R_ID']] = gen_data
			
			#Ivan is in GW
			generators[gen_data['R_ID']]['Existing_Cap_MW'] = float(generators[gen_data['R_ID']]['Existing_Cap_MW']) * 1000.0
			
			#Power to Energy Ratio
			#if generators[gen_data['R_ID']]['Resource'] == 'Hydro Reservoir':
			#	generators[gen_data['R_ID']]['Existing_Cap_MW'] = generators[gen_data['R_ID']]['Existing_Cap_MW'] / generators[gen_data['R_ID']]['Ratio_power_to_energy']
			#if generators[gen_data['R_ID']]['Resource'] == 'Pumped Hydro Storage':
			#	generators[gen_data['R_ID']]['Existing_Cap_MW'] = generators[gen_data['R_ID']]['Existing_Cap_MW'] / generators[gen_data['R_ID']]['Ratio_power_to_energy']
			#if generators[gen_data['R_ID']]['Resource'] == 'Batteries':
			#	generators[gen_data['R_ID']]['Existing_Cap_MW'] = generators[gen_data['R_ID']]['Existing_Cap_MW'] / generators[gen_data['R_ID']]['Ratio_power_to_energy']
				
				
			#Ivan is in GW
			generators[gen_data['R_ID']]['Var_OM_cost_per_MWh'] = float(generators[gen_data['R_ID']]['Var_OM_cost_per_MWh']) * 1000.0
			generators[gen_data['R_ID']]['Heat_rate_MMBTU_per_MWh'] = float(generators[gen_data['R_ID']]['Heat_rate_MMBTU_per_MWh']) * 1000.0
			generators[gen_data['R_ID']]['Start_cost'] = float(generators[gen_data['R_ID']]['Start_cost']) * 1000.0

			
			'Convert to Float'
			generators[gen_data['R_ID']]['Eff_up'] = float(generators[gen_data['R_ID']]['Eff_up'])
			generators[gen_data['R_ID']]['Eff_down'] = float(generators[gen_data['R_ID']]['Eff_down'])
			generators[gen_data['R_ID']]['Hydro_level'] = float(generators[gen_data['R_ID']]['Hydro_level'])
			generators[gen_data['R_ID']]['Ramp_Up_percentage'] = float(generators[gen_data['R_ID']]['Ramp_Up_percentage'])
			generators[gen_data['R_ID']]['Ramp_Dn_percentage'] = float(generators[gen_data['R_ID']]['Ramp_Dn_percentage'])
			generators[gen_data['R_ID']]['Self_disch'] =  float(generators[gen_data['R_ID']]['Self_disch'])
			
			'Convert to Int'
			if generators[gen_data['R_ID']]['Up_time'] > 24:
				generators[gen_data['R_ID']]['Up_time'] = 24
			else:
				generators[gen_data['R_ID']]['Up_time'] = int(generators[gen_data['R_ID']]['Up_time'])
			if generators[gen_data['R_ID']]['Down_time'] > 24:
				generators[gen_data['R_ID']]['Down_time'] = 24
			else:
				generators[gen_data['R_ID']]['Down_time'] = int(generators[gen_data['R_ID']]['Up_time'])
			
			'Startup costs for generators'
			generators[gen_data['R_ID']]['Start_cost'] = generators[gen_data['R_ID']]['Start_cost'] * generators[gen_data['R_ID']]['Existing_Cap_MW']
			
			'Calculate Marginal Cost'
			## VOM[$/MWh] + FuelCost[$/MWh]
			generators[gen_data['R_ID']]['Marginal_Cost'] = (generators[gen_data['R_ID']]['Var_OM_cost_per_MWh'] + 
															 generators[gen_data['R_ID']]['Heat_rate_MMBTU_per_MWh'] * fuels[generators[gen_data['R_ID']]['Fuel']]['Cost_per_MMBtu'])
			
			'Calculate Cluster Size'
			generators[gen_data['R_ID']]['Num_Generators'] = int(generators[gen_data['R_ID']]['Existing_Cap_MW'] / (generators[gen_data['R_ID']]['Cap_size'] * 1000.0 ) )
			if generators[gen_data['R_ID']]['Num_Generators'] == 0:
				generators[gen_data['R_ID']]['Num_Generators'] = 1
			
			'Divide into equal cluster sizes'
			if generators[gen_data['R_ID']]['Num_Generators'] >= GENERATOR_CLUSTER_SIZE and generators[gen_data['R_ID']]['Resource'] in cluster_resources and GENERATOR_CLUSTER_SIZE != 0:
				new_nameplate_capacity = generators[gen_data['R_ID']]['Existing_Cap_MW'] / float(GENERATOR_CLUSTER_SIZE)
				new_num_generators = int(generators[gen_data['R_ID']]['Num_Generators'] / float(GENERATOR_CLUSTER_SIZE))
				
				for cluster_num in range(GENERATOR_CLUSTER_SIZE):
					new_gen_id = int(str(gen_data['R_ID']) + str(cluster_num))
					cluster_generators[new_gen_id] = generators[gen_data['R_ID']].copy()
					cluster_generators[new_gen_id]['Num_Generators'] = new_num_generators
					
					'Set New Startup Cost'
					cluster_generators[new_gen_id]['Start_cost'] = cluster_generators[new_gen_id]['Start_cost'] / GENERATOR_CLUSTER_SIZE
					
					#SYMMETRY PROBLEMS
					cluster_generators[new_gen_id]['Heat_rate_MMBTU_per_MWh'] = float(cluster_generators[new_gen_id]['Heat_rate_MMBTU_per_MWh']) + float(cluster_num) * 55.0
					cluster_generators[new_gen_id]['Marginal_Cost'] = (cluster_generators[new_gen_id]['Var_OM_cost_per_MWh'] + 
																		   cluster_generators[new_gen_id]['Heat_rate_MMBTU_per_MWh'] * fuels[cluster_generators[new_gen_id]['Fuel']]['Cost_per_MMBtu'])
					
					cluster_generators[new_gen_id]['Existing_Cap_MW'] = new_nameplate_capacity + (10.0 * float(cluster_num))
			else:
				#DANGER IF R_ID somehow ends up duplicating with new gen id?
				cluster_generators[gen_data['R_ID']] = generators[gen_data['R_ID']].copy()
			
			
			#print(str(generators[gen_data['R_ID']]['Start_cost']) + generators[gen_data['R_ID']]['Resource'])
			
	#generators is for previous runs before we do the clustering
	#cluster_generators is used in the simulation with our generator clusters
	#both kept for backwards compatability
	
	return generators, cluster_generators
	
def read_india_fuels():
	'fuel_indices, Fuel, Cost_per_MMBtu, CO2_content_tons_perMMBtu'
	
	df = pd.read_csv(INDIA_DATA_DIR + 'Fuels_data.csv')
	fuels = {}
	
	for (index, fuel_data) in df.iterrows():
		fuels[fuel_data['Fuel']] = fuel_data
		fuels[fuel_data['Fuel']]['Cost_per_MMBtu'] = float(fuels[fuel_data['Fuel']]['Cost_per_MMBtu'])
		fuels[fuel_data['Fuel']]['CO2_content_tons_perMMBtu'] = float(fuels[fuel_data['Fuel']]['CO2_content_tons_perMMBtu'])
	return fuels
	
def read_india_network():
	'Names,Share of zonal demand,Network_zones,VRE_Share,DistrZones,CO_2_Max_ton_MWh,'
	'InZoneLossFact_Int,InZoneLossFact_W,InZoneLossFact_I,InZoneLossFact_N,VRE_Share,'
	'Share_in_MV,DistrLossFact_LV_Net_Quad,DistrLossFact_MV_Net_Linear,DistrLossFact_LV_Total_Linear,'
	'Predicted Average Loss at Peak,Assumed Distr. Headroom,Distr_Max_Inject,Distr_Max_Withdraw,'
	'Distr_Inject_Max_Reinforcement_MW,Distr_Withdraw_Max_Reinforcement_MW,Distr_MV_Reinforcement_Cost_per_MW_yr,'
	'Distr_LV_Reinforcement_Cost_per_MW_yr,DistrMarginFact_LV_Linear,DistrMarginFact_LV_Quad,'
	'DistrMarginFact_MV_Linear,DistrMargin_MV_Max,DistrMargin_MV_DiscountFact,Network_lines,'
	'Link_names,z1,z2,z3,z4,z5,Line_Loss_Percentage,Line_Max_Flow_MW,Initial_by_2015,Line_Max_Reinforcement_MW,'
	'Line_Reinforcement_Cost_per_MW_yr,Line_Voltage_kV,Line_Resistance_ohms,Line_X_ohms,'
	'Line_R_ohms,Thetha_max,Peak_Withdrawal_Hours,Peak_Injection_Hours'
	df = pd.read_csv(INDIA_DATA_DIR + 'Network.csv')
	links = {}
	zones = ['1','2','3','4','5']
	
	for (index, network_data) in df.iterrows():
	
		## Locate bus0 and bus1
		bus0 = ''
		bus1 = ''
		
		if int(network_data['z1']) < 0:
			bus0 = 'z1'
		elif int(network_data['z2']) < 0:
			bus0 = 'z2'
		elif int(network_data['z3']) < 0:
			bus0 = 'z3'
		elif int(network_data['z4']) < 0:
			bus0 = 'z4'
		elif int(network_data['z5']) < 0:
			bus0 = 'z5'
		if int(network_data['z1']) > 0:
			bus1 = 'z1'
		elif int(network_data['z2']) > 0:
			bus1 = 'z2'
		elif int(network_data['z3']) > 0:
			bus1 = 'z3'
		elif int(network_data['z4']) > 0:
			bus1 = 'z4'
		elif int(network_data['z5']) > 0:
			bus1 = 'z5'
			
		links[bus0.replace('z',''),bus1.replace('z','')] = network_data
		#Ivan GW conversion
		links[bus0.replace('z',''),bus1.replace('z','')]['Line_Max_Flow_MW'] = float(network_data['Line_Max_Flow_MW']) * 1000.0
	
	return links, zones
	
def read_india_loads():
	'Voll,Demand_segment,Cost_of_demand_curtailment_perMW,'
	'Max_demand_curtailment,Subperiods,Hours_per_period,Sub_Weights,'
	'Time_index,Load_MW_z1,Load_MW_z2,Load_MW_z3,Load_MW_z4,Load_MW_z5'
	
	df = pd.read_csv(INDIA_DATA_DIR + 'Load_data.csv')
	loads = {}
	
	#Unit Conversion from GW to MW
	for (index, load_data) in df.iterrows():
		loads[load_data['Time_index']] = load_data
		loads[load_data['Time_index']]['Load_MW_z1'] = float(load_data['Load_MW_z1']) * 1000.0
		loads[load_data['Time_index']]['Load_MW_z2'] = float(load_data['Load_MW_z2']) * 1000.0
		loads[load_data['Time_index']]['Load_MW_z3'] = float(load_data['Load_MW_z3']) * 1000.0
		loads[load_data['Time_index']]['Load_MW_z4'] = float(load_data['Load_MW_z4']) * 1000.0
		loads[load_data['Time_index']]['Load_MW_z5'] = float(load_data['Load_MW_z5']) * 1000.0
	
	aggregated_loads = []
	not_aggregated_loads = {'1':[],'2':[],'3':[],'4':[],'5':[]}
	
	for hour in loads.keys():
		aggregated_loads.append(loads[hour]['Load_MW_z1'] + 
								loads[hour]['Load_MW_z2'] + 
								loads[hour]['Load_MW_z3'] + 
								loads[hour]['Load_MW_z4'] + 
								loads[hour]['Load_MW_z5'])
		
		not_aggregated_loads['1'].append(loads[hour]['Load_MW_z1'])
		not_aggregated_loads['2'].append(loads[hour]['Load_MW_z2'])
		not_aggregated_loads['3'].append(loads[hour]['Load_MW_z3'])
		not_aggregated_loads['4'].append(loads[hour]['Load_MW_z4'])
		not_aggregated_loads['5'].append(loads[hour]['Load_MW_z5'])
		
	return loads, aggregated_loads, not_aggregated_loads
	
def read_india_gen_variability():
	'Time_index,Solar/z1,Wind/z1,Biomass/z1,Mini Hydro/z1,'
	'Pumped Hydro Storage/z1,Hydro Reservoir/z1,Hydro Run of River/z1,...'
	
	#df = pd.read_csv(INDIA_DATA_DIR + 'Real_Generators_variability.csv')
	df = pd.read_csv(INDIA_DATA_DIR + 'Generators_variability.csv')
	variability = []
	
	for (index, var_data) in df.iterrows():
		variability.append(var_data)
		
	return variability
	
def read_EV_demand_response_bids():
	'Bid_ID,Bidder_ID,zone,Day,start_time,end_time,demand_min_total,'
	'demand_max_total,demand_min_timesteps,demand_max_timesteps'
	df = pd.read_csv(DEMAND_RESPONSE_BIDS)
	#df = pd.read_csv(INDIA_DATA_DIR + 'NO_DR_GENERATED_EV_bids_data.csv')
	
	bids = {}
	for (index, bid_data) in df.iterrows():
	
		min_k_t = bid_data['demand_min_timesteps'].split(';')
		max_k_t = bid_data['demand_max_timesteps'].split(';')
		
		if not len(min_k_t) == len(max_k_t):
			print('EV Bids Format invalid. k_t lens do not equal')
			return None
		
		'Correct bid k format' 
		for i in range(len(min_k_t)):
			min_k_t[i] = float(min_k_t[i].replace(']','').replace('[','').replace(' ','').replace('\n','').replace('\'',''))
			max_k_t[i] = float(max_k_t[i].replace(']','').replace('[','').replace(' ','').replace('\n','').replace('\'',''))
		
		bid_data['demand_min_timesteps'] = min_k_t
		bid_data['demand_max_timesteps'] = max_k_t
			
		
		'Each Bid Data will contain the charging range for the full 24 hours'
		specified_range = range(int(bid_data['start_time']), int(bid_data['end_time']) + 1)
		specified_index = 0
		charging_hours_max = []
		charging_hours_min = []
		
		for hour in range(0,24):
		
			# Will be 0 min and max if not in specified range
			if hour in specified_range:
				charging_hours_min.append(min_k_t[specified_index])
				charging_hours_max.append(max_k_t[specified_index])
				specified_index = specified_index + 1
			else:
				charging_hours_min.append(0.0)
				charging_hours_max.append(0.0)
		
		bid_data['charging_hours_min'] = charging_hours_min
		bid_data['charging_hours_max'] = charging_hours_max
		
		bid_data['demand_min_total'] = float(bid_data['demand_min_total'])
		bid_data['demand_max_total'] = float(bid_data['demand_max_total'])
		bid_data['Day'] = int(bid_data['Day'])
		
		if bid_data['Day'] not in bids.keys():
			bids[bid_data['Day']] = []
			
		bids[bid_data['Day']].append(bid_data)
	
	if len(bids.keys()) == 0:
		return None
		
	return bids
	
def read_resource_perturbations():
	'Resource,Standard Deviation'
	
	try:
		df = pd.read_csv(INDIA_DATA_DIR + 'Resource_perturbations.csv')
	except:
		#print('No Resource_perturbations.csv found in INDIA_DATA_DIR')
		return {}
	
	resource_perturbations = {}
	
	for (index, data) in df.iterrows():
		resource_perturbations[data['Resource']] = {}
		resource_perturbations[data['Resource']]['Standard Deviation'] = float(data['Standard Deviation'])
		resource_perturbations[data['Resource']]['Forced Outage Rate'] = float(data['Forced Outage Rate'])
		
	return resource_perturbations
	
def read_v2g_bids():
	'Bid_ID,Bidder_ID,zone,Day,hour,capacity,cost'
	
	df = pd.read_csv(V2G_BIDS)
	
	bids = {}
	
	for (index, bid_data) in df.iterrows():
		#NEED TO BE CAREFUL READING Bidder_ID CANT HAVE NUMBERS
		
		bid_data["capacity"] = float(bid_data["capacity"])
		bid_data["cost"] = float(bid_data["cost"])
		
		if bid_data['Day'] not in bids.keys():
			bids[bid_data['Day']] = []
		bids[bid_data['Day']].append(bid_data)
		
	return bids
'''
-----------------
'''

'''
Model Components
'''

def demand_perturbations(loads,is_aggregated, time_index, end_time_index, perturbation_level):
	'''
	Returns perturbated loads of 15 minute time steps
	Inputs:
		loads - read_india_loads()
		is_aggregated - True/False of whether or not to consider zones
		time_index - int of starting time_index
		end_time_index - int of end time_index
		perturbation_level - level of std_devs
	Outputs:
		perturbated_15min_loads - randomly perturbated loads with 4 time steps
			σ = std(loads[time_index : end_time_index + 1])
			θ = avg(loads[time_index : end_time_index + 1])
	'''
	
	if is_aggregated:
	
		## Generate perturbated loads for aggregated zone
		demand_mean = np.average( np.array(loads[time_index: (end_time_index + 1)])) 
		demand_std = np.std( np.array(loads[time_index: (end_time_index + 1)]))
		demand_std = 0 #NO DEMAND PERTURBATIONS
		
		perturbated_15min_loads = np.random.normal(loc=demand_mean,scale=perturbation_level*demand_std,size=4)
		
		for load_num in range(4):
			if perturbated_15min_loads[load_num] < 0.0:
					perturbated_15min_loads[load_num] = perturbated_15min_loads[load_num] * -1
	else:
		'''
		## Check Co-Integration between zones
		for i in ['1','2','3','4','5']:
			for k in ['1','2','3','4','5']:
				if not i == k:
					ols_summary, rsquared, t_values, cadf, corr_coef, residuals, beta_param = dicky_fuller_coint(loads[i][0:25],loads[k][0:25])
		'''
		## Generate perturbated loads for each zone
		perturbated_15min_loads = {}
		for zone in ['1','2','3','4','5']:
			demand_mean = np.average( np.array(loads[zone][time_index: (end_time_index + 1)])) 
			demand_std = np.std( np.array(loads[zone][time_index: (end_time_index + 1)]))
			demand_std = 0 #NO DEMAND PERTURBATIONS
			
			perturbated_15min_loads[zone] = np.random.normal(loc=demand_mean,scale=perturbation_level*demand_std,size=4)
			
			for load_num in range(4):
				if perturbated_15min_loads[zone][load_num] < 0.0:
					perturbated_15min_loads[zone][load_num] = perturbated_15min_loads[zone][load_num] * -1
					
	return perturbated_15min_loads

def variability_perturbations(resource,variability, snapshots, perturbation_level, generator_id, generators):
	'''
	Returns a series of perturbations around the variability
	Inputs:
		variability - float
		snapshots - list
		perturbation_level - float
	'''
	
	global GENERATOR_OUTAGE
	global GENERATOR_CLUSTER_OUTAGE
	
	if resource == None:
		resource = 'Default'
		
	# So no peterbations that cause odd solar generation
	if variability == 0:
		return np.array([0] * len(snapshots))
	
	resource_perturbations = read_resource_perturbations()
	length_of_outage = 24
	
	mu = variability
	try:
		sigma = resource_perturbations[resource]['Standard Deviation']
		forced_outrage_rate = float(resource_perturbations[resource]['Forced Outage Rate']) / length_of_outage
	except:
		sigma = 0
		forced_outrage_rate = 0.0
	
	'Forced Outage Randomization'
	random_outage = random.random()
	
	#Clustered
	
	'Check current outages'
	current_total_outage = 0
	restarted_generators = []
	for i in range(len(GENERATOR_CLUSTER_OUTAGE)):
		(cluster_id, num_out, timeleft) = GENERATOR_CLUSTER_OUTAGE[i]
		'Check how many generators are currently out in cluster'
		if cluster_id == generator_id:
			current_total_outage = current_total_outage + num_out
			GENERATOR_CLUSTER_OUTAGE[i] = (cluster_id, num_out, timeleft - 1)
			if ( timeleft - 1 ) == 0:
				restarted_generators.append(i)
	for to_remove in reversed(restarted_generators):
		del GENERATOR_CLUSTER_OUTAGE[to_remove]

	'If there exists active generators test outages'
	current_snapshot_outage = 0 #How many generators in the cluster are out
	if current_total_outage < generators[generator_id]['Num_Generators']:
			
		for i in range(generators[generator_id]['Num_Generators'] - current_total_outage):
			random_outage = random.random()
			if random_outage < forced_outrage_rate and (current_snapshot_outage + current_total_outage) < generators[generator_id]['Num_Generators']:
				current_snapshot_outage = current_snapshot_outage + 1
				
		if current_snapshot_outage > 0:
			outage_data = (generator_id,current_snapshot_outage,24)
			GENERATOR_CLUSTER_OUTAGE.append( outage_data )

	'Calculate current RTM Outage'
	cur_gen_total_outage = 0
	for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
		if cluster_id == generator_id:		
			cur_gen_total_outage = cur_gen_total_outage + num_out
			
	# Return RTM Outage variability
	if cur_gen_total_outage > 0:
		if cur_gen_total_outage > generators[generator_id]['Num_Generators']:
			return np.array([0] * len(snapshots))
		else:
			return np.array([variability - (variability / generators[generator_id]['Num_Generators']) * cur_gen_total_outage ] * len(snapshots))
	
	#Binary
	'''
	if generator_id in GENERATOR_OUTAGE.keys():
		'Outage has already occured'
		GENERATOR_OUTAGE[generator_id] = GENERATOR_OUTAGE[generator_id]  - 1
		if GENERATOR_OUTAGE[generator_id] == 0:
			del GENERATOR_OUTAGE[generator_id]
		return np.array([0] * len(snapshots))
	elif random_outage < forced_outrage_rate:
		'Forced Outage Occurs'
		GENERATOR_OUTAGE[generator_id] = length_of_outage
		GENERATOR_OUTAGE[generator_id] = GENERATOR_OUTAGE[generator_id]  - 1
		return np.array([0] * len(snapshots))
	'''
	
	size = len(snapshots)
	perturbated_rtm_vars = np.random.normal(mu,sigma,size)
	
	'Remove negative and above 1 perturbations'
	for i in range(len(perturbated_rtm_vars)):
		if perturbated_rtm_vars[i] > 1.0:
			perturbated_rtm_vars[i] = 1.0
		if perturbated_rtm_vars[i] < 0.0:
			perturbated_rtm_vars[i] = np.abs(perturbated_rtm_vars[i])
	
	return perturbated_rtm_vars

def Real_Time_Generator_Outage_Variability(resource,variability, snapshots, perturbation_level, generator_id, generators, cur_hour, cur_day):
	'''
	Generator Outage - Average .75
	No Outages - .775
	Outage - .65
	'''
	
	if resource == None:
		resource = 'Default'
		
	# So no peterbations that cause odd solar generation
	if variability == 0:
		return np.array([0] * len(snapshots))
		
	outage_day = 1
	if resource == 'Coal Old' or resource == 'Coal new':
		if outage_day == cur_day:
			return np.array([.75] * len(snapshots))
		else:
			return np.array([.75] * len(snapshots))
	
	return np.array([variability] * len(snapshots))

def calculate_curtailment(network):
	'''
	Calculates the curtailment of renewable energy sources
		Resource = network.generator.carrier
	'''
	
	curtailment = {}
	total_available = {}
	solar_curtailment = {}
	resources = ['Solar','Wind']
	
	'Calculate Curtailment for each resource type'
	for resource in resources:
	
		cur_generators = network.generators.carrier[network.generators.carrier == resource].index
		available_res = [0.0] * len(network.snapshots)
		used_res = [0.0] * len(network.snapshots)
		
		for gen_name in list(cur_generators):
			
			gen_variability = network.generators_t.p_max_pu[gen_name]
			gen_output = network.generators_t.p[gen_name]
			gen_capacity = network.generators.p_nom_max[gen_name]
			if gen_capacity == float("inf"): #Unit Commitment takes different parameters
				gen_capacity = network.generators.p_nom[gen_name]
			
			total_available[gen_name] = []
			curtailment[gen_name] = []
			
			#Curtailment for each snapshot
			for snapshot in network.snapshots:
				total_available[gen_name].append(gen_capacity * gen_variability[snapshot])
				curtailment[gen_name].append(gen_capacity * gen_variability[snapshot] - gen_output[snapshot])
	
	
	#for key in curtailment.keys():
	#	print(str(key) + ' ' + str(curtailment[key]))
		
	#print(total_available)
	
	return curtailment, total_available


def dicky_fuller_coint(series1, series2):
	'''
	Cointegration augmented Dickey Fuller test
	Inputs:
		series1 = array []
		series2 = array []
		Arrays need to be same size
	Returns:
		Ordinary Least Squares Summary
		rsquared
		t_values
		cadf results
		correlation coefficients
	'''
	
	np_series1 = np.array(series1)
	np_series2 = np.array(series2)
	
	## Ordinary Least Squares
	ols_model = sm.OLS(np_series1,np_series2)
	ols_fit = ols_model.fit()
	ols_summary = ols_fit.summary()
	
	rsquared = ols_fit.rsquared
	t_values = ols_fit.tvalues
	
	beta_param = ols_fit.params[0]
	residuals = np_series1  - np.multiply(beta_param, np_series2)
	## Calculating dickey fuller with residuals
	cadf = ts.adfuller(residuals,1)
	corr_coef = np.corrcoef(np_series1, np_series2)[0,1]
	
	return ols_summary, rsquared, t_values, cadf, corr_coef, residuals, beta_param

'''
Optimization Modeling
'''
def add_reserve_attributes():
	'''
	Component: Generator
	Attr Name: reserve_min
	Unit	Type	Default		Description					 Status
	float	MW		0.0			Maximum Reserve Requirement	 Output
	'''
	override_component_attrs["Generator"].loc["reserve_max"] = ["static or series","MW",0.0,"Maximum reserve requirement","Input (optional)"]
	override_component_attrs["Generator"].loc["r"] = ["series","MW",0.0,"Active reserve at bus","Output"]
	
	override_component_attrs["StorageUnit"].loc["reserve_max"] = ["static or series","MW",0.0,"Maximum reserve requirement","Input (optional)"]
	override_component_attrs["StorageUnit"].loc["r"] = ["series","MW",0.0,"Active reserve at bus","Output"]
	
	
def custom_MILP(network,snapshots):
	'''
	All solving cases use this as the "extra_functionality"
	Reserves and Demand Response Added
	'''
	def gen_r_unmet_bounds(model, snapshot):
		return (0,float('inf'))
	def gen_d_unmet_bounds(model,bus_name, snapshot):
		return (0,float('inf'))
		
	network.model.r_unmet = Var(snapshots, domain=NonNegativeReals, bounds=gen_r_unmet_bounds)
	free_pyomo_initializers(network.model.r_unmet)
	
	network.model.d_unmet = Var(list(network.buses.index), snapshots, domain=NonNegativeReals, bounds=gen_d_unmet_bounds)
	free_pyomo_initializers(network.model.d_unmet)
	
	#REDO STORAGE
	if len(snapshots) == 24:
		define_custom_storage_variables_constraints(network,snapshots)
	
	define_slack_nodal_balances(network,snapshots)
	if not DEFAULT_NO_CHARGING:
		define_demand_response_constraints(network,snapshots) #DR
	define_slack_nodal_balance_constraints(network,snapshots)
	
	reserve_constraints(network,snapshots)
	
	if len(snapshots) != 4:
		add_slack_objective(network,snapshots)
	else:
		add_slack_objective(network,snapshots)

def define_custom_storage_variables_constraints(network,snapshots):
	'''
	Modifying opf.py to correctly model hydro reservoirs
	Mostly taken from define_storage_variables_constraints
	'''
	
	sus = network.storage_units
	ext_sus_i = sus.index[sus.p_nom_extendable]
	fix_sus_i = sus.index[~ sus.p_nom_extendable]
	
	model = network.model
	
	## Define storage dispatch variables ##
	
	p_max_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_max_pu', snapshots)
	p_min_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_min_pu', snapshots)

	bounds = {(su,sn) : (0,None) for su in ext_sus_i for sn in snapshots}
	bounds.update({(su,sn) :
					(0,sus.at[su,"p_nom"]*p_max_pu.at[sn, su])
					for su in fix_sus_i for sn in snapshots})
					
	def su_p_dispatch_bounds(model,su_name,snapshot):
		return bounds[su_name,snapshot]
	
	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.storage_p_dispatch)
	network.model.del_component(network.model.storage_p_dispatch_index)
	network.model.del_component(network.model.storage_p_dispatch_index_0)
	network.model.del_component(network.model.storage_p_dispatch_index_1)
	network.model.storage_p_dispatch = Var(list(network.storage_units.index), snapshots,
											domain=NonNegativeReals, bounds=su_p_dispatch_bounds)											
	free_pyomo_initializers(network.model.storage_p_dispatch)
	
	bounds = {(su,sn) : (0,None) for su in ext_sus_i for sn in snapshots}
	bounds.update({(su,sn) :
					(0,-sus.at[su,"p_nom"]*p_min_pu.at[sn, su])
					for su in fix_sus_i
					for sn in snapshots})
					
	def su_p_store_bounds(model,su_name,snapshot):
		return bounds[su_name,snapshot]

	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.storage_p_store)
	network.model.del_component(network.model.storage_p_store_index)
	network.model.del_component(network.model.storage_p_store_index_0)
	network.model.del_component(network.model.storage_p_store_index_1)
	network.model.storage_p_store = Var(list(network.storage_units.index), snapshots,
										domain=NonNegativeReals, bounds=su_p_store_bounds)
	free_pyomo_initializers(network.model.storage_p_store)
	
	## Define spillage variables only for hours with inflow>0. ##
	inflow = get_switchable_as_dense(network, 'StorageUnit', 'inflow', snapshots)
	spill_sus_i = sus.index[inflow.max()>0] #skip storage units without any inflow
	inflow_gt0_b = inflow>0
	spill_bounds = {(su,sn) : (0,inflow.at[sn,su])
					for su in spill_sus_i
					for sn in snapshots
					if inflow_gt0_b.at[sn,su]}
	spill_index = spill_bounds.keys()
	
	def su_p_spill_bounds(model,su_name,snapshot):
		return spill_bounds[su_name,snapshot]

	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.storage_p_spill)
	network.model.del_component(network.model.storage_p_spill_index)
	try:
		network.model.del_component(network.model.storage_p_spill_index_0)
		network.model.del_component(network.model.storage_p_spill_index_1)
	except:
		"storage_p_spill not defined."
	network.model.storage_p_spill = Var(list(spill_index),
										domain=NonNegativeReals, bounds=su_p_spill_bounds)
	free_pyomo_initializers(network.model.storage_p_spill)
	
	## Define generator dispatch constraints for extendable generators ##
	
	def su_p_upper(model,su_name,snapshot):
		return (model.storage_p_dispatch[su_name,snapshot] <=
				model.storage_p_nom[su_name]*p_max_pu.at[snapshot, su_name])
	
	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.storage_p_upper)
	network.model.del_component(network.model.storage_p_upper_index)
	network.model.del_component(network.model.storage_p_upper_index_0)
	network.model.del_component(network.model.storage_p_upper_index_1)
	network.model.storage_p_upper = Constraint(list(ext_sus_i),snapshots,rule=su_p_upper)
	free_pyomo_initializers(network.model.storage_p_upper)

	def su_p_lower(model,su_name,snapshot):
		return (model.storage_p_store[su_name,snapshot] <=
				-model.storage_p_nom[su_name]*p_min_pu.at[snapshot, su_name])

	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.storage_p_lower)
	network.model.del_component(network.model.storage_p_lower_index)
	network.model.del_component(network.model.storage_p_lower_index_0)
	network.model.del_component(network.model.storage_p_lower_index_1)
	network.model.storage_p_lower = Constraint(list(ext_sus_i),snapshots,rule=su_p_lower)
	free_pyomo_initializers(network.model.storage_p_lower)

	## Now define state of charge constraints ##
	
	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.state_of_charge)
	network.model.del_component(network.model.state_of_charge_index)
	network.model.del_component(network.model.state_of_charge_index_0)
	network.model.del_component(network.model.state_of_charge_index_1)
	network.model.state_of_charge = Var(list(network.storage_units.index), snapshots,
										domain=NonNegativeReals, bounds=(0,None))
										
	upper = {(su,sn) : [[(1,model.state_of_charge[su,sn]),
						(-sus.at[su,"max_hours"],model.storage_p_nom[su])],"<=",0.]
			for su in ext_sus_i for sn in snapshots}
	upper.update({(su,sn) : [[(1,model.state_of_charge[su,sn])],"<=",
							sus.at[su,"max_hours"]*sus.at[su,"p_nom"]]
			for su in fix_sus_i for sn in snapshots})

	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.state_of_charge_upper)
	network.model.del_component(network.model.state_of_charge_upper_index)
	network.model.del_component(network.model.state_of_charge_upper_index_0)
	network.model.del_component(network.model.state_of_charge_upper_index_1)
	l_constraint(model, "state_of_charge_upper", upper,
				list(network.storage_units.index), snapshots)

	#this builds the constraint previous_soc + p_store - p_dispatch + inflow - spill == soc
	#it is complicated by the fact that sometimes previous_soc and soc are floats, not variables
	soc = {}
	
	#store the combinations with a fixed soc
	fixed_soc = {}
	
	state_of_charge_set = get_switchable_as_dense(network, 'StorageUnit', 'state_of_charge_set', snapshots)
	
	for su in sus.index:
		for i,sn in enumerate(snapshots):
		
			soc[su,sn] =  [[],"==",0.]
			
			elapsed_hours = network.snapshot_weightings[sn]
			
			if i == 0 and not sus.at[su,"cyclic_state_of_charge"]:
				previous_state_of_charge = sus.at[su,"state_of_charge_initial"]
				soc[su,sn][2] -= ((1-sus.at[su,"standing_loss"])**elapsed_hours
									* previous_state_of_charge)
			else:
				previous_state_of_charge = model.state_of_charge[su,snapshots[i-1]]
				soc[su,sn][0].append(((1-sus.at[su,"standing_loss"])**elapsed_hours,
									previous_state_of_charge))


			state_of_charge = state_of_charge_set.at[sn,su]
			if pd.isnull(state_of_charge):
				state_of_charge = model.state_of_charge[su,sn]
				soc[su,sn][0].append((-1,state_of_charge))
			else:
				soc[su,sn][2] += state_of_charge
				#make sure the variable is also set to the fixed state of charge
				fixed_soc[su,sn] = [[(1,model.state_of_charge[su,sn])],"==",state_of_charge]

			#REMOVE P CHARGE FROM HYDRO RESERVOIR
			if not sus.at[su,"type"] == "Hydro Reservoir":
				soc[su,sn][0].append((sus.at[su,"efficiency_store"]
									* elapsed_hours,model.storage_p_store[su,sn]))
			soc[su,sn][0].append((-(1/sus.at[su,"efficiency_dispatch"]) * elapsed_hours,
								model.storage_p_dispatch[su,sn]))
			soc[su,sn][2] -= inflow.at[sn,su] * elapsed_hours

	for su,sn in spill_index:
		elapsed_hours = network.snapshot_weightings.at[sn]
		storage_p_spill = model.storage_p_spill[su,sn]
		soc[su,sn][0].append((-1.*elapsed_hours,storage_p_spill))
	
	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.state_of_charge_constraint)
	network.model.del_component(network.model.state_of_charge_constraint_index)
	network.model.del_component(network.model.state_of_charge_constraint_index_0)
	network.model.del_component(network.model.state_of_charge_constraint_index_1)
	l_constraint(model,"state_of_charge_constraint",
				soc,list(network.storage_units.index), snapshots)

	#DELETE OLD COMPONENTS
	network.model.del_component(network.model.state_of_charge_constraint_fixed)
	network.model.del_component(network.model.state_of_charge_constraint_fixed_index)
	try:
		network.model.del_component(network.model.state_of_charge_constraint_fixed_index_0)
		network.model.del_component(network.model.state_of_charge_constraint_fixed_index_1)
	except:
		"state_of_charge_constraint_fixed not defined."
	l_constraint(model, "state_of_charge_constraint_fixed",
				fixed_soc, list(fixed_soc.keys()))
	
	
def define_slack_nodal_balances(network,snapshots):
	"""Construct the nodal balance for all elements except the passive
	branches.
	
	Store the nodal balance expression in network._p_balance.
	From PyPSA opf.py
	"""
	
	#dictionary for constraints
	network._p_balance = {(bus,sn) : LExpression()
						  for bus in network.buses.index
						  for sn in snapshots}
						  
	efficiency = get_switchable_as_dense(network, 'Link', 'efficiency', snapshots)

	for cb in network.links.index:
		bus0 = network.links.at[cb,"bus0"]
		bus1 = network.links.at[cb,"bus1"]

		for sn in snapshots:
				network._p_balance[bus0,sn].variables.append((-1,network.model.link_p[cb,sn]))
				network._p_balance[bus1,sn].variables.append((efficiency.at[sn,cb],network.model.link_p[cb,sn]))

	#Add any other buses to which the links are attached
	for i in [int(col[3:]) for col in network.links.columns if col[:3] == "bus" and col not in ["bus0","bus1"]]:
		efficiency = get_switchable_as_dense(network, 'Link', 'efficiency{}'.format(i), snapshots)
		for cb in network.links.index[network.links["bus{}".format(i)] != ""]:
			bus = network.links.at[cb, "bus{}".format(i)]
			for sn in snapshots:
				network._p_balance[bus,sn].variables.append((efficiency.at[sn,cb],network.model.link_p[cb,sn]))
	
	added_buses = []
	
	for gen in network.generators.index:
		bus = network.generators.at[gen,"bus"]
		sign = network.generators.at[gen,"sign"]
		for sn in snapshots:
			network._p_balance[bus,sn].variables.append((sign,network.model.generator_p[gen,sn]))
			
		#Tony Modification
		'Add Unmet Demand Variable at each Bus and Snapshot'
		## UNMET DEMAND B,T
		if bus not in added_buses:
			added_buses.append(bus)
			for sn in snapshots:
				network._p_balance[bus,sn].variables.append((1,network.model.d_unmet[bus,sn]))
		#
		
	load_p_set = get_switchable_as_dense(network, 'Load', 'p_set', snapshots)
	for load in network.loads.index:
		bus = network.loads.at[load,"bus"]
		sign = network.loads.at[load,"sign"]
		for sn in snapshots:
			network._p_balance[bus,sn].constant += sign*load_p_set.at[sn,load]
			
	for su in network.storage_units.index:
		bus = network.storage_units.at[su,"bus"]
		sign = network.storage_units.at[su,"sign"]
		for sn in snapshots:
				network._p_balance[bus,sn].variables.append((sign,network.model.storage_p_dispatch[su,sn]))
				network._p_balance[bus,sn].variables.append((-sign,network.model.storage_p_store[su,sn]))
				
	for store in network.stores.index:
		bus = network.stores.at[store,"bus"]
		sign = network.stores.at[store,"sign"]
		for sn in snapshots:
			network._p_balance[bus,sn].variables.append((sign,network.model.store_p[store,sn]))


def define_slack_nodal_balance_constraints(network,snapshots):
	''' From PyPSA opf.py '''
	passive_branches = network.passive_branches()
	
	for branch in passive_branches.index:
		bus0 = passive_branches.at[branch,"bus0"]
		bus1 = passive_branches.at[branch,"bus1"]
		bt = branch[0]
		bn = branch[1]
		for sn in snapshots:
			network._p_balance[bus0,sn].variables.append((-1,network.model.passive_branch_p[bt,bn,sn]))
			network._p_balance[bus1,sn].variables.append((1,network.model.passive_branch_p[bt,bn,sn]))
	power_balance = {k: LConstraint(v,"==",LExpression()) for k,v in iteritems(network._p_balance)}
	
	## DELETE ALL EXISTING POWER BALANCE CONSTRAINTS
	network.model.del_component(network.model.power_balance)
	network.model.del_component(network.model.power_balance_index)
	network.model.del_component(network.model.power_balance_index_0)
	network.model.del_component(network.model.power_balance_index_1)
	#
	
	l_constraint(network.model, "power_balance", power_balance, list(network.buses.index), snapshots)

def define_demand_response_constraints(network,snapshots):

	'''
	Include a Demand Response Portion into the nodal balances
	
	Example
	P_1 + P_2 = D + d
	
	d_1 + d_2 = d_total
	DR_k_t <= d <= DR_k_t
	DR_min_total <= d_total <= DR_max_total
	
	-DR_ramp_down <= d <= DR_ramp_up
	
	'''
	
	'Ensures Inputs are correct for Demand Response'
	all_EV_bids = read_EV_demand_response_bids()
	if all_EV_bids is None :
		return
	if not len(snapshots) == 24 and not len(snapshots) == 4:
		return
		
	#cur_day = int(CURRENT_TIME_INDEX / 24) + 1 # Need to increment the Global CURRENT_TIME_INDEX
	cur_day = (int(CURRENT_TIME_INDEX / 24) % 7 + 1)  # Days 1 - 7 Only in bids
	is_aggregated = False

	try:
		current_EV_bids = all_EV_bids[cur_day]
	except:
		return
	
	if len(list(network.buses.index)) == 1:
		is_aggregated = True
		main_bus = list(network.buses.index)[0] # Set as main bus
	
	#Find the zones that have bidding
	bid_zones = []
	for i in range(len(current_EV_bids)):
		if str(current_EV_bids[i]['zone']) not in bid_zones:
			bid_zones.append(str(current_EV_bids[i]['zone']))
	if is_aggregated:
		bid_zones = [main_bus]

	#RTM Demand Response
	''' RTM Demand Response Curtailment '''
	if len(snapshots) == 4:
		define_rtm_demand_response_constraints(network,snapshots,bid_zones,current_EV_bids,cur_day,is_aggregated)
		return
	''' ------------------------------- '''

	'''Define Variables'''
	def ev_dr_bid_bounds(model,bid_index,bus,snapshot):
		'''Sets the bounds for individual bids by k,b,t'''
		return (model.current_EV_bids[bid_index]['charging_hours_min'][snapshot],model.current_EV_bids[bid_index]['charging_hours_max'][snapshot])
	def dr_total_bounds(model,bus,snapshot):
		return (0,float('inf'))
	network.model.current_EV_bids = current_EV_bids
	network.model.demand_response_dDR_bid = Var(range(len(current_EV_bids)), bid_zones,snapshots, domain=Reals,bounds=ev_dr_bid_bounds)
	network.model.demand_response_dTotalDR = Var(bid_zones,snapshots, domain=Reals, bounds=dr_total_bounds)
	
	'''POWER BALANCE'''
	'Modify the Power Balance Constraint with Demand Response'
	for sn in snapshots:
		if is_aggregated == False:
			for bus in bid_zones:
				network._p_balance[bus,sn].variables.append((-1,network.model.demand_response_dTotalDR[bus,sn]))
		else:
			network._p_balance[main_bus,sn].variables.append((-1,network.model.demand_response_dTotalDR[main_bus,sn]))
	
	'''TOTAL CONSTRAINTS'''
	'Add Demand Response min and max constraints'
	ev_dr_total_lower = {}
	ev_dr_total_upper = {}
	
	if is_aggregated == False:
		'NOT AGGREGATED'
		ev_dr_total_lower_constraint = {}
		ev_dr_total_upper_constraint = {}
		for i in range(len(current_EV_bids)):
			
			cur_bus = str(current_EV_bids[i]['zone'])
			ev_dr_total_lower_constraint[i] = [[]]
			ev_dr_total_upper_constraint[i] = [[]]
			
			for sn in snapshots:
				ev_dr_total_lower_constraint[i][0].append((1,network.model.demand_response_dDR_bid[i,cur_bus,sn]))
				ev_dr_total_upper_constraint[i][0].append((1,network.model.demand_response_dDR_bid[i,cur_bus,sn]))
				
			ev_dr_total_lower_constraint[i].append('>=')
			ev_dr_total_lower_constraint[i].append(current_EV_bids[i]['demand_min_total'])
			ev_dr_total_upper_constraint[i].append('<=')
			ev_dr_total_upper_constraint[i].append(current_EV_bids[i]['demand_max_total'])

		l_constraint(network.model, "ev_dr_bids_lower_total", ev_dr_total_lower_constraint,range(len(current_EV_bids)))
		l_constraint(network.model, "ev_dr_bids_upper_total", ev_dr_total_upper_constraint,range(len(current_EV_bids)))
		
	else:
		'AGGREGATED'
		ev_dr_total_lower_constraint = {}
		ev_dr_total_upper_constraint = {}
		for i in range(len(current_EV_bids)):
			
			ev_dr_total_lower_constraint[i] = [[]]
			ev_dr_total_upper_constraint[i] = [[]]
			
			for sn in snapshots:
				ev_dr_total_lower_constraint[i][0].append((1,network.model.demand_response_dDR_bid[i,main_bus,sn]))
				ev_dr_total_upper_constraint[i][0].append((1,network.model.demand_response_dDR_bid[i,main_bus,sn]))
				
			ev_dr_total_lower_constraint[i].append('>=')
			ev_dr_total_lower_constraint[i].append(current_EV_bids[i]['demand_min_total'])
			ev_dr_total_upper_constraint[i].append('<=')
			ev_dr_total_upper_constraint[i].append(current_EV_bids[i]['demand_max_total'])

		l_constraint(network.model, "ev_dr_bids_lower_total", ev_dr_total_lower_constraint,range(len(current_EV_bids)))
		l_constraint(network.model, "ev_dr_bids_upper_total", ev_dr_total_upper_constraint,range(len(current_EV_bids)))
		
	'''
	if is_aggregated == False:
		for bus in bid_zones:
			ev_dr_total_lower[bus] = 0.0
			ev_dr_total_upper[bus] = 0.0
		for i in range(len(current_EV_bids)):
			ev_dr_total_lower[current_EV_bids[i]['zone']] = ev_dr_total_lower[current_EV_bids[i]['zone']] + current_EV_bids[i]['demand_min_total']
			ev_dr_total_upper[current_EV_bids[i]['zone']] = ev_dr_total_upper[current_EV_bids[i]['zone']] + current_EV_bids[i]['demand_max_total']
			
		ev_dr_total_lower_eq = {(bus,sn) :
						[ [ev_dr_total_lower[bus]],
						"<=",[(1,network.model.demand_response_dTotalDR[bus,sn])]]
						for bus in bid_zones
						for sn in snapshots}
		ev_dr_total_upper_eq = {(bus,sn) :
						[[(1,network.model.demand_response_dTotalDR[bus,sn])],
						"<=", [ev_dr_total_upper[bus]]]
						for bus in bid_zones
						for sn in snapshots}
		l_constraint(network.model, "ev_dr_bids_lower_total", ev_dr_total_lower_eq, bid_zones, snapshots)
		l_constraint(network.model, "ev_dr_bids_upper_total", ev_dr_total_upper_eq, bid_zones, snapshots)
		
	else:
	
		ev_dr_total_lower[main_bus] = 0.0
		ev_dr_total_upper[main_bus] = 0.0

		for i in range(len(current_EV_bids)):
			ev_dr_total_lower[main_bus] = ev_dr_total_lower[main_bus] + current_EV_bids[i]['demand_min_total']
			ev_dr_total_upper[main_bus] = ev_dr_total_upper[main_bus] + current_EV_bids[i]['demand_max_total']

		ev_dr_total_lower_eq = {(sn) :
						[[(1,network.model.demand_response_dTotalDR[main_bus,sn])],
						">=", ev_dr_total_lower[main_bus]]
						for sn in snapshots}
		ev_dr_total_upper_eq = {(sn) :
						[[(1,network.model.demand_response_dTotalDR[main_bus,sn])],
						"<=", ev_dr_total_upper[main_bus]]
						for sn in snapshots}
						
		
		l_constraint(network.model, "ev_dr_bids_lower_total", ev_dr_total_lower_eq, snapshots)
		l_constraint(network.model, "ev_dr_bids_upper_total", ev_dr_total_upper_eq, snapshots)
	'''	
	
	'''TotalDR Equality Constraint'''
	total_constraint = {}
	
	for sn in snapshots:
		if is_aggregated == False:
			total_DR_eq = [[]]
			for bus in bid_zones:
				for i in range(len(current_EV_bids)):
					if bus == str(current_EV_bids[i]['zone']):
						total_DR_eq[0].append((1,network.model.demand_response_dDR_bid[i,bus,sn]))
						
				total_DR_eq[0].append((-1,network.model.demand_response_dTotalDR[bus,sn]))
			total_DR_eq.append("==")
			total_DR_eq.append(0.0)
			
			'''
			for bus in bid_zones:
				
				total_DR_eq = [[]]
				for i in range(len(current_EV_bids)):
					cur_bus = str(current_EV_bids[i]['zone'])
					if cur_bus == bus:
						total_DR_eq[0].append((1,network.model.demand_response_dDR_bid[i,bus,sn]))
				
				if(len(total_DR_eq[0])):
					total_DR_eq[0] = [0.0]
					
				total_DR_eq.append("==")
				total_DR_eq.append((1,network.model.demand_response_dTotalDR[bus,sn]))
				print(total_DR_eq)
				#l_constraint(network.model, "ev_dr_bids_total_equality", total_DR_eq, range(len(current_EV_bids)), bid_zones, snapshots)
			'''
		else:
			
			total_DR_eq = [[]]
			for i in range(len(current_EV_bids)):
				total_DR_eq[0].append((1,network.model.demand_response_dDR_bid[i,main_bus,sn]))
			total_DR_eq[0].append((-1,network.model.demand_response_dTotalDR[main_bus,sn]))
			total_DR_eq.append("==")
			total_DR_eq.append(0.0)
			
		total_constraint[sn] = total_DR_eq
	
	l_constraint(network.model, "ev_dr_bids_total_equality", total_constraint, snapshots)
	#print(network.model.ev_dr_bids_total_equality.pprint())

def define_rtm_demand_response_constraints(network,snapshots,bid_zones,current_EV_bids,cur_day, is_aggregated):
	'''
	If variability causes problems, give up demand response
	'''
	
	'''Define Variables'''
	def dr_curtailment_bounds(model,bus,snapshot):
		return (0,DAILY_FIXED_DR[(bus,CURRENT_HOUR_INDEX)])
	network.model.demand_response_dCurtailDR = Var(bid_zones,snapshots, domain=Reals, bounds=dr_curtailment_bounds)
	
	
	'''Curtailment Constraints'''
	'Limits Curtailment in each 15 minute block to sum of the hourly fixed Demand Response'
	'''
	total_upper_curtail_constraint = {}
	total_lower_curtail_constraint = {}
	for bus in bid_zones:
		bus_upper_curtail_constraint = [[]]
		bus_lower_curtail_constraint = [[]]
		
		for sn in snapshots:
			bus_upper_curtail_constraint[0].append((1,network.model.demand_response_dCurtailDR[bus,sn]))
			bus_lower_curtail_constraint[0].append((1,network.model.demand_response_dCurtailDR[bus,sn]))
			
		bus_upper_curtail_constraint.append("<=")
		bus_upper_curtail_constraint.append(DAILY_FIXED_DR[(bus,CURRENT_HOUR_INDEX)])
		
		bus_lower_curtail_constraint.append(">=")
		bus_lower_curtail_constraint.append(0.0)
		
		total_upper_curtail_constraint[bus] = bus_upper_curtail_constraint
		total_lower_curtail_constraint[bus] = bus_lower_curtail_constraint
	l_constraint(network.model, "demand_response_curtail_upper",total_upper_curtail_constraint, bid_zones)
	l_constraint(network.model, "demand_response_curtail_lower",total_lower_curtail_constraint, bid_zones)
	'''
	
	'''Modify the Objective'''
	model = network.model
	
	## Delete Default Objective
	model.del_component(model.objective)
	##
	
	extendable_generators = network.generators[network.generators.p_nom_extendable]
	ext_sus = network.storage_units[network.storage_units.p_nom_extendable]
	ext_stores = network.stores[network.stores.e_nom_extendable]
	passive_branches = network.passive_branches()

	extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]
	extendable_links = network.links[network.links.p_nom_extendable]
	
	suc_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable & (network.generators.start_up_cost > 0)]
	sdc_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable & (network.generators.shut_down_cost > 0)]
	
	marginal_cost_it = zip(get_switchable_as_iter(network, 'Generator', 'marginal_cost', snapshots),
						   get_switchable_as_iter(network, 'StorageUnit', 'marginal_cost', snapshots),
						   get_switchable_as_iter(network, 'Store', 'marginal_cost', snapshots),
						   get_switchable_as_iter(network, 'Link', 'marginal_cost', snapshots))

	objective = LExpression()
	
	for sn, marginal_cost in zip(snapshots, marginal_cost_it):
		gen_mc, su_mc, st_mc, link_mc = marginal_cost
		
		weight = network.snapshot_weightings[sn]
		for gen in network.generators.index:
			coefficient = gen_mc.at[gen] * weight
			objective.variables.extend([(coefficient, model.generator_p[gen, sn])])
			
		for su in network.storage_units.index:
			coefficient = su_mc.at[su] * weight
			objective.variables.extend([(coefficient, model.storage_p_dispatch[su,sn])])
			
		for store in network.stores.index:
			coefficient = st_mc.at[store] * weight
			objective.variables.extend([(coefficient, model.store_p[store,sn])])
			
		for link in network.links.index:
			coefficient = link_mc.at[link] * weight
			objective.variables.extend([(coefficient, model.link_p[link,sn])])

	'Adding Unmet Reserve to Objective'
	## Unmet Reserve Default = $1000
	for sn in snapshots:
		objective.variables.extend([(CRESERVE, model.r_unmet[sn])])
	
	'Adding Unmet Demand to Objective'
	## Unmet Demand Default = $9000
	for bus in network.buses.index:
		for sn in snapshots:
			objective.variables.extend([(CVOLL, model.d_unmet[bus,sn])])
	
	#NB: for capital costs we subtract the costs of existing infrastructure p_nom/s_nom
	objective.variables.extend([(extendable_generators.at[gen,"capital_cost"], model.generator_p_nom[gen])
								for gen in extendable_generators.index])
	objective.constant -= (extendable_generators.capital_cost * extendable_generators.p_nom).zsum()
	
	objective.variables.extend([(ext_sus.at[su,"capital_cost"], model.storage_p_nom[su])
								for su in ext_sus.index])
								
	objective.constant -= (ext_sus.capital_cost*ext_sus.p_nom).zsum()
	
	objective.variables.extend([(ext_stores.at[store,"capital_cost"], model.store_e_nom[store])
								for store in ext_stores.index])
	objective.constant -= (ext_stores.capital_cost*ext_stores.e_nom).zsum()
	
	objective.variables.extend([(extendable_passive_branches.at[b,"capital_cost"], model.passive_branch_s_nom[b])
								for b in extendable_passive_branches.index])
	objective.constant -= (extendable_passive_branches.capital_cost * extendable_passive_branches.s_nom).zsum()
	
	objective.variables.extend([(extendable_links.at[b,"capital_cost"], model.link_p_nom[b])
								for b in extendable_links.index])
	
	objective.constant -= (extendable_links.capital_cost * extendable_links.p_nom).zsum()

	'''Modify Power Balance and Objective'''
	'Modify the Power Balance Constraint with Demand Response Curtailment'
	for sn in snapshots:
		if is_aggregated == False:
			for bus in bid_zones:
				network._p_balance[bus,sn].variables.append((1,network.model.demand_response_dCurtailDR[bus,sn]))
				objective.variables.extend([(CDRCURTAIL,network.model.demand_response_dCurtailDR[bus,sn])])
		else:
			network._p_balance[main_bus,sn].variables.append((1,network.model.demand_response_dCurtailDR[main_bus,sn]))
			objective.variables.extend([(CDRCURTAIL,network.model.demand_response_dCurtailDR[main_bus,sn])])
			
	## Unit commitment costs

	objective.variables.extend([(1, model.generator_start_up_cost[gen,sn]) for gen in suc_gens_i for sn in snapshots])
	objective.variables.extend([(1, model.generator_shut_down_cost[gen,sn]) for gen in sdc_gens_i for sn in snapshots])
	
	l_objective(model,objective)
	

	
def reserve_constraints(network,snapshots):
	'''
	Set correct constraints for Co-Optimization with reserves
	
	Example
	P_1 + P_2 = Demand 
	R_1 + R_2 >= Min_System_Reserve

	0<= P_1 + R_1 <= Max_nom_P_1
	0<= P_2 + R_2 <= Max_nom_P_2
	0<= R_1 <= Max_nom_R1
	0<= R_2 <= Max_nom_R2
	'''
	
	def gen_p_bounds_f(model,gen_name,snapshot):
		return gen_p_bounds[gen_name,snapshot]
		
	def gen_r_nom_bounds(model, gen_name,snapshot):
		return (0,network.generators.at[gen_name,"reserve_max"])
	
	def stor_r_nom_bounds(model, stor_name, snapshot):
		return (0,network.storage_units.at[stor_name,"reserve_max"])
	
	#Taken from PyPSA opf.py
	extendable_gens_i = network.generators.index[network.generators.p_nom_extendable]
	fixed_gens_i = network.generators.index[~network.generators.p_nom_extendable & ~network.generators.committable]
	fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]
	
	
	p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu', snapshots)
	p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu', snapshots)
	r_max_pu = get_switchable_as_dense(network, 'Generator', 'reserve_max',snapshots)
	r_max_pu_iter = get_switchable_as_iter(network, 'Generator', 'reserve_max',snapshots)
	
	gen_p_bounds = {(gen,sn) : (None,None)
					for gen in extendable_gens_i | fixed_committable_gens_i
					for sn in snapshots}
	
	if len(fixed_gens_i):
		var_lower = p_min_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])
		var_upper = p_max_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])
		gen_p_bounds.update({(gen,sn) : (var_lower[gen][sn],var_upper[gen][sn])
							for gen in fixed_gens_i
							for sn in snapshots})
							
	## Define Reserve Dispatch variables
	#network.model.generator_r = Var(list(network.generators.index), snapshots, domain=Reals, bounds=gen_p_bounds_f)
	network.model.generator_r = Var(list(network.generators.index), snapshots, domain=NonNegativeReals, bounds=gen_r_nom_bounds)
	free_pyomo_initializers(network.model.generator_r)
	
	network.model.generator_r_nom = Var(list(network.generators.index), snapshots, domain=NonNegativeReals, bounds=gen_r_nom_bounds)
	free_pyomo_initializers(network.model.generator_r)
	
	
	## Define Capacity Constraint for Storage Units
	sus = network.storage_units
	fixed_sus_i = sus.index[~ sus.p_nom_extendable] # Only committable storage units allowed
	stor_p_max_pu = get_switchable_as_dense(network, 'StorageUnit', 'p_max_pu', snapshots)
	
	network.model.storage_units_r = Var(list(fixed_sus_i),snapshots,domain=NonNegativeReals,bounds=stor_r_nom_bounds)
	free_pyomo_initializers(network.model.storage_units_r)
	
	'Equation 1j'
	stor_p_r_upper_soc = {(stor,sn) :
						[[(1,network.model.storage_p_dispatch[stor,sn]), 						#qDischarge
						(1,network.model.storage_units_r[stor,sn]),		 						#qRes
						#(-1,np.minimum(network.model.state_of_charge[stor,sn],network.storage_units.p_nom[stor]))
						(-1,network.model.state_of_charge[stor,sn])
						],
						"<=",0.]
						for stor in list(fixed_sus_i) for sn in snapshots}
	l_constraint(network.model,"stor_p_r_upper_soc",stor_p_r_upper_soc, list(fixed_sus_i),snapshots)
	
	stor_p_r_upper_dis = {(stor,sn) :
						[[(1,network.model.storage_p_dispatch[stor,sn]), 						#qDischarge
						(1,network.model.storage_units_r[stor,sn]),		 						#qRes
						],
						"<=",stor_p_max_pu.at[sn,stor]*network.storage_units.p_nom[stor]]
						for stor in list(fixed_sus_i) for sn in snapshots}
						
	l_constraint(network.model,"stor_p_r_upper_dis",stor_p_r_upper_dis, list(fixed_sus_i),snapshots)
	
	## Define Capacity Constraint P + R <= u*P_max]
	
	'Equation 5b'
	gen_p_r_upper = {(gen,sn) :
					[[(1,network.model.generator_p[gen,sn]),
					(1,network.model.generator_r[gen,sn]),
					(-p_max_pu.at[sn, gen]*network.generators.p_nom[gen],network.model.generator_status[gen,sn])
					],
					"<=",0.]
					for gen in fixed_committable_gens_i for sn in snapshots}
	l_constraint(network.model, "generator_p_r_upper", gen_p_r_upper, list(fixed_committable_gens_i), snapshots)
	gen_r_max = {(gen,sn) :
					[[(1,network.model.generator_r[gen,sn]),
					(-p_max_pu.at[sn, gen]*network.generators.p_nom[gen],network.model.generator_status[gen,sn])
					],
					"<=",0.]
					for gen in fixed_committable_gens_i for sn in snapshots}
					
	l_constraint(network.model, "generator_r_max", gen_r_max, list(fixed_committable_gens_i), snapshots)
	
	
	'''
	## Define Reserve Constraint 0 <= R <= R_max
	gen_r_lower = {(gen,sn) :
                   [[(1,network.model.generator_r[gen,sn]),],">=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
	l_constraint(network.model, "generator_r_lower", gen_r_lower, list(extendable_gens_i), snapshots)
	gen_r_upper = {(gen,sn) :
                   [[(1,network.model.generator_r[gen,sn]), (-1,network.model.generator_r_nom[gen]) ],"<=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
	l_constraint(network.model, "generator_r_upper", gen_r_upper, list(extendable_gens_i), snapshots)
	'''
	## Define System Reserve Constraint R1 + ... + RN >= R_min
	# Similiar to global constraints PyPSA does for CO2 emissions
	snapshot_ctr = 0
	global_constraints = {}
	for snap in list(r_max_pu_iter):
	
		reserve_global_eq = [[]]
		for gen,max_res in snap.items():
			reserve_global_eq[0].append((1,network.model.generator_r[gen,snapshot_ctr]))
		
		#Storage Units as reserve
		for stor in list(fixed_sus_i):
			reserve_global_eq[0].append((1,network.model.storage_units_r[stor,snapshot_ctr]))
		
		## UNMET RESERVE
		reserve_global_eq[0].append((1,network.model.r_unmet[snapshot_ctr]))
		##
		
		reserve_global_eq.append(">=")
		reserve_global_eq.append(MIN_SYS_RESERVE_ENERGY_MW)
		
		global_constraints[snapshot_ctr] = reserve_global_eq
		snapshot_ctr = snapshot_ctr + 1
	
	l_constraint(network.model, "system_r_req", global_constraints,snapshots)
	
def add_slack_objective(network,snapshots):
	'''Modify the Objective'''
	model = network.model
	
	## Delete Default Objective
	model.del_component(model.objective)
	##
	
	extendable_generators = network.generators[network.generators.p_nom_extendable]
	ext_sus = network.storage_units[network.storage_units.p_nom_extendable]
	ext_stores = network.stores[network.stores.e_nom_extendable]
	passive_branches = network.passive_branches()

	extendable_passive_branches = passive_branches[passive_branches.s_nom_extendable]
	extendable_links = network.links[network.links.p_nom_extendable]
	
	suc_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable & (network.generators.start_up_cost > 0)]
	sdc_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable & (network.generators.shut_down_cost > 0)]
	
	marginal_cost_it = zip(get_switchable_as_iter(network, 'Generator', 'marginal_cost', snapshots),
						   get_switchable_as_iter(network, 'StorageUnit', 'marginal_cost', snapshots),
						   get_switchable_as_iter(network, 'Store', 'marginal_cost', snapshots),
						   get_switchable_as_iter(network, 'Link', 'marginal_cost', snapshots))

	objective = LExpression()
	
	for sn, marginal_cost in zip(snapshots, marginal_cost_it):
		gen_mc, su_mc, st_mc, link_mc = marginal_cost
		
		weight = network.snapshot_weightings[sn]
		for gen in network.generators.index:
			coefficient = gen_mc.at[gen] * weight
			objective.variables.extend([(coefficient, model.generator_p[gen, sn])])
			
		for su in network.storage_units.index:
			coefficient = su_mc.at[su] * weight
			objective.variables.extend([(coefficient, model.storage_p_dispatch[su,sn])])
			
		for store in network.stores.index:
			coefficient = st_mc.at[store] * weight
			objective.variables.extend([(coefficient, model.store_p[store,sn])])
			
		for link in network.links.index:
			coefficient = link_mc.at[link] * weight
			objective.variables.extend([(coefficient, model.link_p[link,sn])])

	'Adding Unmet Reserve to Objective'
	## Unmet Reserve Default = $1000
	for sn in snapshots:
		objective.variables.extend([(CRESERVE, model.r_unmet[sn])])
	
	'Adding Unmet Demand to Objective'
	## Unmet Demand Default = $9000
	for bus in network.buses.index:
		for sn in snapshots:
			objective.variables.extend([(CVOLL, model.d_unmet[bus,sn])])
	
	#NB: for capital costs we subtract the costs of existing infrastructure p_nom/s_nom
	objective.variables.extend([(extendable_generators.at[gen,"capital_cost"], model.generator_p_nom[gen])
								for gen in extendable_generators.index])
	objective.constant -= (extendable_generators.capital_cost * extendable_generators.p_nom).zsum()
	
	objective.variables.extend([(ext_sus.at[su,"capital_cost"], model.storage_p_nom[su])
								for su in ext_sus.index])
								
	objective.constant -= (ext_sus.capital_cost*ext_sus.p_nom).zsum()
	
	objective.variables.extend([(ext_stores.at[store,"capital_cost"], model.store_e_nom[store])
								for store in ext_stores.index])
	objective.constant -= (ext_stores.capital_cost*ext_stores.e_nom).zsum()
	
	objective.variables.extend([(extendable_passive_branches.at[b,"capital_cost"], model.passive_branch_s_nom[b])
								for b in extendable_passive_branches.index])
	objective.constant -= (extendable_passive_branches.capital_cost * extendable_passive_branches.s_nom).zsum()
	
	objective.variables.extend([(extendable_links.at[b,"capital_cost"], model.link_p_nom[b])
								for b in extendable_links.index])
	
	objective.constant -= (extendable_links.capital_cost * extendable_links.p_nom).zsum()

	## Unit commitment costs

	objective.variables.extend([(1, model.generator_start_up_cost[gen,sn]) for gen in suc_gens_i for sn in snapshots])
	objective.variables.extend([(1, model.generator_shut_down_cost[gen,sn]) for gen in sdc_gens_i for sn in snapshots])
	
	l_objective(model,objective)

	
def UC_security_constraints(network,snapshots):
	'''
	Set correct constraints for Co-Optimization with reserves
	
	Example
	P_1 + P_2 = Demand 
	R_1 + R_2 >= Min_System_Reserve

	0<= P_1 + R_1 <= Max_nom_P_1
	0<= P_2 + R_2 <= Max_nom_P_2
	0<= R_1 <= Max_nom_R1
	0<= R_2 <= Max_nom_R2
	'''
	
	def gen_p_bounds_f(model,gen_name,snapshot):
		return gen_p_bounds[gen_name,snapshot]
		
	def gen_r_nom_bounds(model, gen_name,snapshot):
		return (0,network.generators.at[gen_name,"reserve_max"])
	
	#Taken from PyPSA opf.py
	extendable_gens_i = network.generators.index[network.generators.p_nom_extendable]
	fixed_gens_i = network.generators.index[~network.generators.p_nom_extendable & ~network.generators.committable]
	fixed_committable_gens_i = network.generators.index[~network.generators.p_nom_extendable & network.generators.committable]
	
	p_min_pu = get_switchable_as_dense(network, 'Generator', 'p_min_pu', snapshots)
	p_max_pu = get_switchable_as_dense(network, 'Generator', 'p_max_pu', snapshots)
	r_max_pu = get_switchable_as_dense(network, 'Generator', 'reserve_max',snapshots)
	r_max_pu_iter = get_switchable_as_iter(network, 'Generator', 'reserve_max',snapshots)
	
	gen_p_bounds = {(gen,sn) : (None,None)
					for gen in extendable_gens_i | fixed_committable_gens_i
					for sn in snapshots}
	
	if len(fixed_gens_i):
		var_lower = p_min_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])
		var_upper = p_max_pu.loc[:,fixed_gens_i].multiply(network.generators.loc[fixed_gens_i, 'p_nom'])
		gen_p_bounds.update({(gen,sn) : (var_lower[gen][sn],var_upper[gen][sn])
							for gen in fixed_gens_i
							for sn in snapshots})
							
	## Define Reserve Dispatch variables
	#network.model.generator_r = Var(list(network.generators.index), snapshots, domain=Reals, bounds=gen_p_bounds_f)
	network.model.generator_r = Var(list(network.generators.index), snapshots, domain=NonNegativeReals, bounds=gen_r_nom_bounds)
	free_pyomo_initializers(network.model.generator_r)
	
	network.model.generator_r_nom = Var(list(network.generators.index), snapshots, domain=NonNegativeReals, bounds=gen_r_nom_bounds)
	free_pyomo_initializers(network.model.generator_r)
	
	## Define Capacity Constraint P + R <= P_max
	gen_p_r_upper = {(gen,sn) :
					[[(1,network.model.generator_p[gen,sn]),
					(1,network.model.generator_r[gen,sn]),
					(-p_max_pu.at[sn, gen],network.model.generator_p_nom[gen])],
					"<=",0.]
					for gen in extendable_gens_i for sn in snapshots}
	l_constraint(network.model, "generator_p_r_upper", gen_p_r_upper, list(extendable_gens_i), snapshots)
	'''
	## Define Reserve Constraint 0 <= R <= R_max
	gen_r_lower = {(gen,sn) :
                   [[(1,network.model.generator_r[gen,sn]),],">=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
	l_constraint(network.model, "generator_r_lower", gen_r_lower, list(extendable_gens_i), snapshots)
	gen_r_upper = {(gen,sn) :
                   [[(1,network.model.generator_r[gen,sn]), (-1,network.model.generator_r_nom[gen]) ],"<=",0.]
                   for gen in extendable_gens_i for sn in snapshots}
	l_constraint(network.model, "generator_r_upper", gen_r_upper, list(extendable_gens_i), snapshots)
	'''
	## Define System Reserve Constraint R1 + ... + RN >= R_min
	snapshot_ctr = 0
	global_constraints = {}
	for snap in list(r_max_pu_iter):
	
		reserve_global_eq = [[]]
		for gen,max_res in snap.items():
			reserve_global_eq[0].append((1,network.model.generator_r[gen,snapshot_ctr]))
		
		
		reserve_global_eq.append(">=")
		reserve_global_eq.append(MIN_SYS_RESERVE_ENERGY_MW)
		
		global_constraints[snapshot_ctr] = reserve_global_eq
		snapshot_ctr = snapshot_ctr + 1
	
	l_constraint(network.model, "system_r_req", global_constraints,snapshots)
	
'''
-----------------
'''

'''
Build India Network
'''	
def add_india_loads(network,zones,is_aggregated,loads):
	'''
	Adds Specified loads to the buses
	'''
	if is_aggregated:
		network.add("Load", "{} load".format('India'), bus='India', p_set=loads)
	else:
		for zone in zones:
			network.add("Load", "Zone {} load".format(zone), bus=zone, p_set=loads[zone])
	
	return network
	
def ED_build_india_generators(network,generators,india_tranmission,zones,is_aggregated, variability, time_index, perturbate_variability):
	
	'''
	Inputs:
		network - PyPSA network object
		generators - read_india_generators()
		india_tranmission - read_india_network()
		zones - read_india_network()
		is_aggregated - true/false for whether to use all 5 zones or one whole
	Output:
		returns the PyPSA network object with generators added
	'''
	
	if not is_aggregated:
		'NOT AGGREGATED'
		
		## Add each zone as a bus
		for zone in zones:
			network.add('Bus',zone)
			
		## Add each generator and storage unit to each zone
		for gen_id in generators.keys():
			
			if int(generators[gen_id]['STOR']) == 1:
				network.add("StorageUnit",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								efficiency_store = generators[gen_id]['Eff_up'],
								efficiency_dispatch = generators[gen_id]['Eff_down'],
								state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
								standing_loss = generators[gen_id]['Self_disch']
							)
			else:
				
				'Variability Added'
				try:
				
					
					variability_key = str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])
					
					'RTM Variability perturbations'
					gen_var = variability[time_index][variability_key]
					if perturbate_variability:
						perturbated_vars = variability_perturbations(generators[gen_id]['Resource'],variability[time_index][variability_key], network.snapshots, 1, gen_id, generators)
						gen_var = perturbated_vars
					
					#print(gen_var)
					#print(generators[gen_id]['Existing_Cap_MW'])
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = gen_var
								)
				
				except KeyError:
					print('Error has occured in reading RTM variability.')
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = 1.0
								)
		## Add transmission lines
		for (zone0,zone1) in india_tranmission.keys():
		
			## Bi-directional and lossless links
			network.add("Link", 
						"{} - {}".format(zone0,zone1),
						bus0=zone0, bus1=zone1,
						marginal_cost=0.0,
						efficiency=1,
						p_nom=india_tranmission[(zone0,zone1)]['Line_Max_Flow_MW'],
						p_min_pu=-1,
						)
	else:
		'AGGREGATED'
		
		network.add('Bus','India')
		
		## Add each generator and storage unit to each zone
		for gen_id in generators.keys():
			if int(generators[gen_id]['STOR']) == 1:
				network.add("StorageUnit",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								efficiency_store = generators[gen_id]['Eff_up'],
								efficiency_dispatch = generators[gen_id]['Eff_down'],
								state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
								standing_loss = generators[gen_id]['Self_disch']
							)
			else:
				'''
				network.add("Generator",
							"{}".format(gen_id),
							bus = 'India',
							carrier = generators[gen_id]['Resource'],
							p_nom_extendable = True,
							p_nom_max = generators[gen_id]['Existing_Cap_MW'],
							p_nom_min = 0,
							marginal_cost = generators[gen_id]['Marginal_Cost'],
							reserve_max = generators[gen_id]['Existing_Cap_MW']
							)
				'''
				
				'Variability Added'
				try:
				
					variability_key = str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])
					
					'RTM Variability perturbations'
					gen_var = variability[time_index][variability_key]
					if perturbate_variability:
						perturbated_vars = variability_perturbations(generators[gen_id]['Resource'],variability[time_index][variability_key], network.snapshots, 1, gen_id, generators)
						gen_var = perturbated_vars
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = gen_var
								)
				
				except KeyError:
					print('Error has occured in reading RTM variability.')
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = 1.0
								)
			
	return network

def UC_build_india_generators(network,generators,india_tranmission,zones,is_aggregated,variability, time_index):
	'''
	Inputs:
		network - PyPSA network object
		generators - read_india_generators()
		india_tranmission - read_india_network()
		zones - read_india_network()
		is_aggregated - true/false for whether to use all 5 zones or one whole
	Output:
		returns the PyPSA network object with generators added for unit commitment
	'''
	
	if not is_aggregated:
		'NOT AGGREGATED'
		
		## Add each zone as a bus
		for zone in zones:
			network.add('Bus',zone)
			
		## Add each generator to each zone
		for gen_id in generators.keys():
			if int(generators[gen_id]['Commit']) == 1:
				
				'Variability Added'
				try:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					outage_day_var = np.copy(np.array(day_variability))
					
					'Planning Known Outages'
					if gen_id in GENERATOR_OUTAGE.keys():
						for out_hr in range(GENERATOR_OUTAGE[gen_id]):
							day_variability[out_hr] = 0
					
					
					'Calculate Cluster Outages'
					current_total_outage = 0
					for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
						if cluster_id == gen_id:
							for out_hr in range(timeleft):
								outage_day_var[out_hr] = outage_day_var[out_hr] - float(num_out * day_variability[out_hr] / (generators[gen_id]['Num_Generators'] - current_total_outage ))
								
							current_total_outage = current_total_outage + num_out
							

					#Apply the outages to variability
					if not current_total_outage == 0:
						#print("GEN " + str(gen_id) + " :" + str(current_total_outage))
						#print(outage_day_var)
						day_variability = outage_day_var
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								start_up_cost = float(generators[gen_id]['Start_cost']),
								committable=True,
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								initial_status = 1,
								p_min_pu = float(generators[gen_id]['Min_power']),
								p_max_pu = day_variability,
								ramp_limit_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_down = generators[gen_id]['Ramp_Dn_percentage'],
								ramp_limit_start_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_shut_down = generators[gen_id]['Ramp_Dn_percentage'],
								min_up_time = generators[gen_id]['Up_time'],
								min_down_time = generators[gen_id]['Down_time']
								)
				except KeyError:
					#Newer data files does not trigger this
					print('Error in defining variability')
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								start_up_cost = float(generators[gen_id]['Start_cost']),
								committable=True,
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								initial_status = 0,
								p_min_pu = float(generators[gen_id]['Min_power']),
								p_max_pu = 1.0,
								ramp_limit_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_down = generators[gen_id]['Ramp_Dn_percentage'],
								ramp_limit_start_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_shut_down = generators[gen_id]['Ramp_Dn_percentage'],
								min_up_time = generators[gen_id]['Up_time'],
								min_down_time = generators[gen_id]['Down_time']
								)
				
			else:
				# Add Uncommittable storage units and generators to network
				
				if int(generators[gen_id]['HYDRO']) == 1:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					day_inflow = np.multiply(np.multiply(day_variability, generators[gen_id]['Existing_Cap_MW']), (1 / generators[gen_id]['Ratio_power_to_energy']) )
					
					
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									#efficiency_store = 0,
									#efficiency_dispatch = 10000000000,
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									inflow = day_inflow,
									max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']),
									cyclic_state_of_charge = True,
									p_max_pu = generators[gen_id]['Eff_down'],
									type=generators[gen_id]['Resource']
								)
					
				elif int(generators[gen_id]['STOR']) == 1:
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']) ,
									cyclic_state_of_charge = True,
									p_max_pu = generators[gen_id]['Eff_down']
								)
				else:
					'Adding Variable Renewable Energy generators'
					#Try adding Variability for VRE
					try:
						day_variability = []
						for vars in variability[time_index:time_index + 24]:
							#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
							day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
							
						'Planning Forced Outages'
						if gen_id in GENERATOR_OUTAGE.keys():
							for out_hr in range(GENERATOR_OUTAGE[gen_id]):
								day_variability[out_hr] = 0
						
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									committable = False,
									reserve_max = 0.0,
									p_max_pu = day_variability
									)
				
					except KeyError:
						print('No variability added for VRE gen')
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									committable = False,
									reserve_max = 0.0,
									p_max_pu = 1.0
									)
			
		## Add transmission lines
		for (zone0,zone1) in india_tranmission.keys():
		
			## Bi-directional and lossless links
			network.add("Link", 
						"{} - {}".format(zone0,zone1),
						bus0=zone0, bus1=zone1,
						marginal_cost=0.0,
						efficiency=1,
						p_nom=india_tranmission[(zone0,zone1)]['Line_Max_Flow_MW'],
						p_min_pu=-1)
	else:
		'AGGREGATED'
		
		network.add('Bus','India')
		## Add each generator to each zone
		for gen_id in generators.keys():
			if int(generators[gen_id]['Commit']) == 1:
				
				'''
				network.add("Generator",
							"{} {}".format(generators[gen_id]['Resource'], gen_id),
							bus = 'India',
							carrier = generators[gen_id]['Resource'],
							p_nom = generators[gen_id]['Existing_Cap_MW'],
							marginal_cost = generators[gen_id]['Marginal_Cost'],
							start_up_cost = float(generators[gen_id]['Start_cost']),
							committable=True,
							reserve_max = generators[gen_id]['Existing_Cap_MW'],
							initial_status = 0,
							p_min_pu=float(generators[gen_id]['Min_power'])
				)
				'''
				
				'Variability Added'
				try:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					outage_day_var = np.copy(np.array(day_variability))
					
					'Planning Forced Outages'
					if gen_id in GENERATOR_OUTAGE.keys():
						for out_hr in range(GENERATOR_OUTAGE[gen_id]):
							day_variability[out_hr] = 0
					
					'Calculate Cluster Outages'
					current_total_outage = 0
					for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
						if cluster_id == gen_id:
							for out_hr in range(timeleft):
								outage_day_var[out_hr] = outage_day_var[out_hr] - float(num_out * day_variability[out_hr] / (generators[gen_id]['Num_Generators'] - current_total_outage ))
								
							current_total_outage = current_total_outage + num_out
							
					#Apply the outages to variability
					if not current_total_outage == 0:
						day_variability = outage_day_var
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								carrier = generators[gen_id]['Resource'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								start_up_cost = float(generators[gen_id]['Start_cost']),
								committable=True,
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								initial_status = 0,
								p_min_pu = float(generators[gen_id]['Min_power']),
								p_max_pu = day_variability,
								ramp_limit_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_down = generators[gen_id]['Ramp_Dn_percentage'],
								ramp_limit_start_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_shut_down = generators[gen_id]['Ramp_Dn_percentage'],
								min_up_time = generators[gen_id]['Up_time'],
								min_down_time = generators[gen_id]['Down_time']
								)
				
				except KeyError:
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								carrier = generators[gen_id]['Resource'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								start_up_cost = float(generators[gen_id]['Start_cost']),
								committable=True,
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								initial_status = 0,
								p_min_pu = float(generators[gen_id]['Min_power']),
								p_max_pu = 1.0,
								ramp_limit_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_down = generators[gen_id]['Ramp_Dn_percentage'],
								ramp_limit_start_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_shut_down = generators[gen_id]['Ramp_Dn_percentage'],
								min_up_time = generators[gen_id]['Up_time'],
								min_down_time = generators[gen_id]['Down_time']
								)
								
				
			else:
				'''
				if int(generators[gen_id]['HYDRO']) == 1:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					day_inflow = np.multiply(day_variability, generators[gen_id]['Existing_Cap_MW'])
					
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									inflow = day_inflow,
									p_max_pu = generators[gen_id]['Ratio_power_to_energy']
								)
				'''
				if int(generators[gen_id]['STOR']) == 1:
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = 'India',
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									#p_max_pu = generators[gen_id]['Ratio_power_to_energy']
								)
					
				else:
					'Variability Added'
					try:
						day_variability = []
						for vars in variability[time_index:time_index + 24]:
							#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
							day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
							
						'Planning Forced Outages'
						if gen_id in GENERATOR_OUTAGE.keys():
							for out_hr in range(GENERATOR_OUTAGE[gen_id]):
								day_variability[out_hr] = 0
								
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = 'India',
									carrier = generators[gen_id]['Resource'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									committable = False,
									reserve_max = 0.0,
									p_max_pu = day_variability
									)
				
					except KeyError:	
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = 'India',
									carrier = generators[gen_id]['Resource'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									committable = False,
									reserve_max = 0.0,
									p_max_pu = 1.0
									)
			
	return network


def Fixed_ED_build_india_generators(network,generators,india_tranmission,zones,is_aggregated,variability, time_index,generator_status):
	'''
	DAM SCED with fixed generator statuses to find LMPs
	'''
	if not is_aggregated:
		'NOT AGGREGATED'
		
		## Add each zone as a bus
		for zone in zones:
			network.add('Bus',zone)

		## Add each generator and storage unit to each zone if committed
		for gen_id in generators.keys():
		
			generator_name = "{} {}".format(generators[gen_id]['Resource'], gen_id)	
			if not ( int(generators[gen_id]['STOR']) == 1 or int(generators[gen_id]['HYDRO']) == 1 ):
			#if not ( int(generators[gen_id]['STOR']) == 1):
				'Variability Added'
				try:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					outage_day_var = np.copy(np.array(day_variability))
					
					'Planning Forced Outages'
					if gen_id in GENERATOR_OUTAGE.keys():
						for out_hr in range(GENERATOR_OUTAGE[gen_id]):
							day_variability[out_hr] = 0
					
					'Calculate Cluster Outages'
					current_total_outage = 0
					for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
						if cluster_id == gen_id:
							for out_hr in range(timeleft):
								outage_day_var[out_hr] = outage_day_var[out_hr] - float(num_out * day_variability[out_hr] / (generators[gen_id]['Num_Generators'] - current_total_outage ))
								
							current_total_outage = current_total_outage + num_out
							
					#Apply the outages to variability
					if not current_total_outage == 0:
						day_variability = outage_day_var
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max =  generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = np.multiply( day_variability, np.array(generator_status[generator_name]))
								)
					
				except KeyError:
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = np.array(generator_status[generator_name])
								)
			else:
				# Add Uncommittable storage units
				
				if int(generators[gen_id]['HYDRO']) == 1:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					day_inflow = np.multiply(np.multiply(day_variability, generators[gen_id]['Existing_Cap_MW']), (1 / generators[gen_id]['Ratio_power_to_energy']) )
					
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									#efficiency_store = 0,
									#efficiency_dispatch = 10000000000,
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									inflow = day_inflow,
									max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']) ,
									cyclic_state_of_charge = True,
									p_max_pu = generators[gen_id]['Eff_down'],
									type=generators[gen_id]['Resource']
								)
					
				else:
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									max_hours = int(int(1 / generators[gen_id]['Ratio_power_to_energy']) / 2),
									cyclic_state_of_charge = True,
									p_max_pu = generators[gen_id]['Eff_down']
							)
							
		## Add transmission lines
		for (zone0,zone1) in india_tranmission.keys():
		
			## Bi-directional and lossless links
			network.add("Link", 
						"{} - {}".format(zone0,zone1),
						bus0=zone0, bus1=zone1,
						marginal_cost=0.0,
						efficiency=1,
						p_nom=india_tranmission[(zone0,zone1)]['Line_Max_Flow_MW'],
						p_min_pu=-1,
						)
	else:
		'AGGREGATED'
		network.add('Bus','India')
		
		## Add each generator and storage unit
		for gen_id in generators.keys():
		
			generator_name = "{} {}".format(generators[gen_id]['Resource'], gen_id)	
			#if not (int(generators[gen_id]['STOR']) == 1 or int(generators[gen_id]['HYDRO']) == 1):
			if not (int(generators[gen_id]['STOR']) == 1):
				'Variability Added'
				try:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					outage_day_var = np.copy(np.array(day_variability))
					
					'Planning Forced Outages'
					if gen_id in GENERATOR_OUTAGE.keys():
						for out_hr in range(GENERATOR_OUTAGE[gen_id]):
							day_variability[out_hr] = 0
					
					'Calculate Cluster Outages'
					current_total_outage = 0
					for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
						if cluster_id == gen_id:
							for out_hr in range(timeleft):
								outage_day_var[out_hr] = outage_day_var[out_hr] - float(num_out * day_variability[out_hr] / (generators[gen_id]['Num_Generators'] - current_total_outage ))
								
							current_total_outage = current_total_outage + num_out
							
					#Apply the outages to variability
					if not current_total_outage == 0:
						day_variability = outage_day_var
					
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max =  generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = np.multiply( day_variability, np.array(generator_status[generator_name]))
								)
					
				except KeyError:
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = 'India',
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = np.array(generator_status[generator_name])
								)
			else:
				# Add Uncommittable storage units
				'''
				if int(generators[gen_id]['HYDRO']) == 1:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					day_inflow = np.multiply(day_variability, generators[gen_id]['Existing_Cap_MW'])
					
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									inflow = day_inflow,
									p_max_pu = day_variability
								)
				'''
				#else:
				network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = 'India',
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch']
								)
							
	return network

#######################
# STORAGE STATUS KEPT #
#######################

def Day_Ahead_Unit_Commitment(network,generators,india_tranmission,zones,is_aggregated,variability, time_index, previous_storage_soc=None):
	'''
	Inputs:
		network - PyPSA network object
		generators - read_india_generators()
		india_tranmission - read_india_network()
		zones - read_india_network()
		is_aggregated - true/false for whether to use all 5 zones or one whole
		variability - the read_india_gen_variability()
		time_index - starting time index
		previous_storage_soc - from previous DAM SCUC to set initial storage levels
	Output:
		returns the PyPSA network object with generators added for unit commitment
	'''
	
	#Used for MIN_SYS_RESERVE_ENERGY_MW
	VRE_Reserve = 0
	
	#Will always run not aggregated case
	if not is_aggregated:
		'NOT AGGREGATED'
		
		## Add each zone as a bus
		for zone in zones:
			network.add('Bus',zone)
			
		## Add each generator to each zone
		for gen_id in generators.keys():
			if int(generators[gen_id]['Commit']) == 1:
				
				'Variability Added'
				try:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					outage_day_var = np.copy(np.array(day_variability))
					
					'Planning Known Outages'
					#if gen_id in GENERATOR_OUTAGE.keys():
					#	for out_hr in range(GENERATOR_OUTAGE[gen_id]):
					#		day_variability[out_hr] = 0
					
					
					'Calculate Cluster Outages'
					current_total_outage = 0
					for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
						if cluster_id == gen_id:
							for out_hr in range(timeleft):
								outage_day_var[out_hr] = outage_day_var[out_hr] - float(num_out * day_variability[out_hr] / (generators[gen_id]['Num_Generators'] - current_total_outage ))
								
							current_total_outage = current_total_outage + num_out
							

					#Apply the outages to variability
					if not current_total_outage == 0:
						#print("GEN " + str(gen_id) + " :" + str(current_total_outage))
						#print(outage_day_var)
						#day_variability = outage_day_var
						day_variability = day_variability #DOES NOT FACTOR IN OUTAGES
					
					#Apply Reserve Maximum constraints
					reserve_coefficient = 1.0
					if generators[gen_id]['Resource'] == 'CCGT Old' or generators[gen_id]['Resource'] == 'CCGT New' or generators[gen_id]['Resource'] == 'Nuclear Old' or generators[gen_id]['Resource'] == 'Nuclear New':
						reserve_coefficient = 0.75
					elif generators[gen_id]['Resource'] == 'Coal Old' or generators[gen_id]['Resource'] == 'Coal New' or generators[gen_id]['Resource'] == 'Biomass':
						reserve_coefficient = 0.60
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								start_up_cost = float(generators[gen_id]['Start_cost']),
								committable=True,
								reserve_max = reserve_coefficient * generators[gen_id]['Existing_Cap_MW'],
								initial_status = 1,
								p_min_pu = float(generators[gen_id]['Min_power']),
								p_max_pu = day_variability,
								ramp_limit_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_down = generators[gen_id]['Ramp_Dn_percentage'],
								ramp_limit_start_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_shut_down = generators[gen_id]['Ramp_Dn_percentage'],
								min_up_time = generators[gen_id]['Up_time'],
								min_down_time = generators[gen_id]['Down_time']
								)
				except KeyError:
					#Newer data files does not trigger this
					print('Error in defining variability')
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom = generators[gen_id]['Existing_Cap_MW'],
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								start_up_cost = float(generators[gen_id]['Start_cost']),
								committable=True,
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								initial_status = 0,
								p_min_pu = float(generators[gen_id]['Min_power']),
								p_max_pu = 1.0,
								ramp_limit_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_down = generators[gen_id]['Ramp_Dn_percentage'],
								ramp_limit_start_up = generators[gen_id]['Ramp_Up_percentage'],
								ramp_limit_shut_down = generators[gen_id]['Ramp_Dn_percentage'],
								min_up_time = generators[gen_id]['Up_time'],
								min_down_time = generators[gen_id]['Down_time']
								)
				
			else:
				# Add Uncommittable storage units and generators to network
				
				if int(generators[gen_id]['HYDRO']) == 1:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					day_inflow = np.multiply(np.multiply(day_variability, generators[gen_id]['Existing_Cap_MW']), (1 / generators[gen_id]['Ratio_power_to_energy']) )
					
					max_energy = int(1 / generators[gen_id]['Ratio_power_to_energy']) * generators[gen_id]['Existing_Cap_MW']
					
					if previous_storage_soc is None:
						network.add("StorageUnit",
										"{} {}".format(generators[gen_id]['Resource'], gen_id),
										bus = generators[gen_id]['zone'],
										p_nom = generators[gen_id]['Existing_Cap_MW'],
										efficiency_store = generators[gen_id]['Eff_up'],
										efficiency_dispatch = generators[gen_id]['Eff_down'],
										state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
										standing_loss = generators[gen_id]['Self_disch'],
										inflow = day_inflow,
										max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']),
										cyclic_state_of_charge = False,
										p_max_pu = generators[gen_id]['Eff_down'],
										type=generators[gen_id]['Resource'],
										reserve_max = generators[gen_id]['Existing_Cap_MW']
									)
					else:
						
						#Some Scenarios have storage units with 0 Existing Capacity
						if max_energy == 0:
							current_initial_state_ratio = 0
						else:
							#if max_energy is 0 then division by 0 error
							current_initial_state_ratio = previous_storage_soc[generators[gen_id]['Resource'] + ' ' +str(gen_id)][len(network.snapshots) - 1] / max_energy
						
						network.add("StorageUnit",
										"{} {}".format(generators[gen_id]['Resource'], gen_id),
										bus = generators[gen_id]['zone'],
										p_nom = generators[gen_id]['Existing_Cap_MW'],
										efficiency_store = generators[gen_id]['Eff_up'],
										efficiency_dispatch = generators[gen_id]['Eff_down'],
										state_of_charge_initial = current_initial_state_ratio * generators[gen_id]['Existing_Cap_MW'],
										standing_loss = generators[gen_id]['Self_disch'],
										inflow = day_inflow,
										max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']),
										cyclic_state_of_charge = False,
										p_max_pu = generators[gen_id]['Eff_down'],
										type=generators[gen_id]['Resource'],
										reserve_max = generators[gen_id]['Existing_Cap_MW']
									)
									
						
				elif int(generators[gen_id]['STOR']) == 1:
					
					max_energy = int(1 / generators[gen_id]['Ratio_power_to_energy']) * generators[gen_id]['Existing_Cap_MW']
					
					if previous_storage_soc is None:
						network.add("StorageUnit",
										"{} {}".format(generators[gen_id]['Resource'], gen_id),
										bus = generators[gen_id]['zone'],
										p_nom = generators[gen_id]['Existing_Cap_MW'],
										efficiency_store = generators[gen_id]['Eff_up'],
										efficiency_dispatch = generators[gen_id]['Eff_down'],
										state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
										standing_loss = generators[gen_id]['Self_disch'],
										max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']) ,
										cyclic_state_of_charge = False,
										p_max_pu = generators[gen_id]['Eff_down'],
										reserve_max = generators[gen_id]['Existing_Cap_MW']
									)
					else:
					
						#Some Scenarios have storage units with 0 Existing Capacity
						if max_energy == 0:
							current_initial_state_ratio = 0
						else:
							#if max_energy is 0 then division by 0 error
							current_initial_state_ratio = previous_storage_soc[generators[gen_id]['Resource'] + ' ' +str(gen_id)][len(network.snapshots) - 1] / max_energy
						network.add("StorageUnit",
										"{} {}".format(generators[gen_id]['Resource'], gen_id),
										bus = generators[gen_id]['zone'],
										p_nom = generators[gen_id]['Existing_Cap_MW'],
										efficiency_store = generators[gen_id]['Eff_up'],
										efficiency_dispatch = generators[gen_id]['Eff_down'],
										state_of_charge_initial = current_initial_state_ratio * generators[gen_id]['Existing_Cap_MW'],
										standing_loss = generators[gen_id]['Self_disch'],
										max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']) ,
										cyclic_state_of_charge = False,
										p_max_pu = generators[gen_id]['Eff_down'],
										reserve_max = generators[gen_id]['Existing_Cap_MW']
									)
				else:
					'Adding Variable Renewable Energy generators'
					#Try adding Variability for VRE
					try:
						day_variability = []
						for vars in variability[time_index:time_index + 24]:
							#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
							day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
							
						'Planning Forced Outages'
						if gen_id in GENERATOR_OUTAGE.keys():
							for out_hr in range(GENERATOR_OUTAGE[gen_id]):
								day_variability[out_hr] = 0
						
						#Used for MIN_SYS_RESERVE_ENERGY_MW
						VRE_Reserve = np.average(np.multiply(0.005,(np.multiply(generators[gen_id]['Existing_Cap_MW'],day_variability))))
						
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									committable = False,
									reserve_max = 0.0,
									p_max_pu = day_variability
									)
				
					except KeyError:
						print('No variability added for VRE gen')
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									committable = False,
									reserve_max = 0.0,
									p_max_pu = 1.0
									)
			
		## Add transmission lines
		for (zone0,zone1) in india_tranmission.keys():
		
			## Bi-directional and lossless links
			network.add("Link", 
						"{} - {}".format(zone0,zone1),
						bus0=zone0, bus1=zone1,
						marginal_cost=0.0,
						efficiency=1,
						p_nom=india_tranmission[(zone0,zone1)]['Line_Max_Flow_MW'],
						p_min_pu=-1)
	else:
		print('AGGREGATE RUN. WILL BREAK')
		
	return network, VRE_Reserve

def Day_Ahead_Economic_Dispatch(network,generators,india_tranmission,zones,is_aggregated,variability, time_index,generator_status,previous_storage_soc):
	
	'''
	DAM SCED with fixed generator statuses - find LMPs
	Inputs:
	Outputs:
	
	'''
	if not is_aggregated:
		'NOT AGGREGATED'
		
		## Add each zone as a bus
		for zone in zones:
			network.add('Bus',zone)

		## Add each generator and storage unit to each zone if committed
		for gen_id in generators.keys():
		
			generator_name = "{} {}".format(generators[gen_id]['Resource'], gen_id)	
			if not ( int(generators[gen_id]['STOR']) == 1 or int(generators[gen_id]['HYDRO']) == 1 ):
			#if not ( int(generators[gen_id]['STOR']) == 1):
				'Variability Added'
				try:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						#day_variability.append(vars[str(generators[gen_id]['Resource']) + '/z' + str(generators[gen_id]['zone'])])
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					outage_day_var = np.copy(np.array(day_variability))
					
					'Planning Forced Outages'
					#if gen_id in GENERATOR_OUTAGE.keys():
					#	for out_hr in range(GENERATOR_OUTAGE[gen_id]):
					#		day_variability[out_hr] = 0
					
					'Calculate Cluster Outages'
					current_total_outage = 0
					for (cluster_id, num_out, timeleft) in GENERATOR_CLUSTER_OUTAGE:
						if cluster_id == gen_id:
							for out_hr in range(timeleft):
								outage_day_var[out_hr] = outage_day_var[out_hr] - float(num_out * day_variability[out_hr] / (generators[gen_id]['Num_Generators'] - current_total_outage ))
								
							current_total_outage = current_total_outage + num_out
							
					#Apply the outages to variability
					if not current_total_outage == 0:
						#day_variability = outage_day_var
						day_variability = day_variability #DOES NOT FACTOR IN OUTAGES
					
					reserve_coefficient = 1.0
					if generators[gen_id]['Resource'] == 'CCGT Old' or generators[gen_id]['Resource'] == 'CCGT New' or generators[gen_id]['Resource'] == 'Nuclear Old' or generators[gen_id]['Resource'] == 'Nuclear New':
						reserve_coefficient = 0.75
					elif generators[gen_id]['Resource'] == 'Coal Old' or generators[gen_id]['Resource'] == 'Coal New' or generators[gen_id]['Resource'] == 'Biomass':
						reserve_coefficient = 0.60
					
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max =  generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = reserve_coefficient * generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = np.multiply( day_variability, np.array(generator_status[generator_name]))
								)
					
				except KeyError:
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = np.array(generator_status[generator_name])
								)
			else:
				# Add Uncommittable storage units
				
				if int(generators[gen_id]['HYDRO']) == 1:
					day_variability = []
					for vars in variability[time_index:time_index + 24]:
						day_variability.append(vars[str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])])
						
					day_inflow = np.multiply(np.multiply(day_variability, generators[gen_id]['Existing_Cap_MW']), (1 / generators[gen_id]['Ratio_power_to_energy']) )
					
					max_energy = int(1 / generators[gen_id]['Ratio_power_to_energy']) * generators[gen_id]['Existing_Cap_MW']
					
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									inflow = day_inflow,
									max_hours = int(1 / generators[gen_id]['Ratio_power_to_energy']) ,
									cyclic_state_of_charge = True,
									p_max_pu = generators[gen_id]['Eff_down'],
									type=generators[gen_id]['Resource']
								)
					
				else:
					network.add("StorageUnit",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									p_nom = generators[gen_id]['Existing_Cap_MW'],
									efficiency_store = generators[gen_id]['Eff_up'],
									efficiency_dispatch = generators[gen_id]['Eff_down'],
									state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
									standing_loss = generators[gen_id]['Self_disch'],
									max_hours = int(int(1 / generators[gen_id]['Ratio_power_to_energy']) / 2),
									cyclic_state_of_charge = True,
									p_max_pu = generators[gen_id]['Eff_down'],
									reserve_max = generators[gen_id]['Existing_Cap_MW']
							)
							
		## Add transmission lines
		for (zone0,zone1) in india_tranmission.keys():
		
			## Bi-directional and lossless links
			network.add("Link", 
						"{} - {}".format(zone0,zone1),
						bus0=zone0, bus1=zone1,
						marginal_cost=0.0,
						efficiency=1,
						p_nom=india_tranmission[(zone0,zone1)]['Line_Max_Flow_MW'],
						p_min_pu=-1,
						)
	else:
		print('AGGREGATE RUN. WILL BREAK')
		
	return network


	
def Real_Time_Economic_Dispatch(network,generators,india_tranmission,zones,is_aggregated, variability, time_index, perturbate_variability, 
								v2g_participation, cur_hour, cur_day, dam_stor_pow, dam_stor_soc, dam_coal_nuc_pow):
	
	'''
	Inputs:
		network - PyPSA network object
		generators - read_india_generators()
		india_tranmission - read_india_network()
		zones - read_india_network()
		is_aggregated - true/false for whether to use all 5 zones or one whole
		variability - read_india_gen_variability()
		time_index - current time index
		perturbate_variability - boolean to perturbate variability
		v2g - boolean to add v2g participation
	Output:
		returns the PyPSA network object with generators added
	'''
	
	Inflexible_Resources = ['Coal Old','Coal new', 'Nuclear New', 'Nuclear Old'] #Generators like Nuclear and Coal that must be on due to shutdown costs and lower ramp rates
	
	if not is_aggregated:
		'NOT AGGREGATED'
		
		## Add each zone as a bus
		for zone in zones:
			network.add('Bus',zone)
			
		## Add each generator and storage unit to each zone
		for gen_id in generators.keys():
			
			if int(generators[gen_id]['STOR']) == 1:
				
				'Uses DAM Storage scheduling to match'
				if dam_stor_pow["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour] < 0:
					stor_p_min_pu = -1
					stor_p_max_pu = 0
					stor_initial = 0
					#stor_initial = dam_stor_soc["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour] - dam_stor_pow["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour]
					stor_p_nom = dam_stor_pow["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour]
					stor_p_store = stor_initial
					#stor_charge_end = [0,0,0,dam_stor_soc["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour]]
				else:
					stor_p_min_pu = 0
					stor_p_max_pu = 1
					stor_initial = dam_stor_pow["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour]
					if stor_initial > 0:
						stor_p_max_pu = .25
						
					stor_p_nom = dam_stor_pow["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour]
					stor_p_store = 0
					#stor_charge_end = [0,0,0,dam_stor_soc["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour]]
				#stor_initial = 0
				
				network.add("StorageUnit",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								efficiency_store = generators[gen_id]['Eff_up'],
								efficiency_dispatch = generators[gen_id]['Eff_down'],
								standing_loss = generators[gen_id]['Self_disch'],
								p_nom = generators[gen_id]['Existing_Cap_MW'] * 4,
								p_min_pu = stor_p_min_pu,
								p_max_pu = stor_p_max_pu,
								state_of_charge_initial = stor_initial * 4,
								p_store = stor_p_store
								#state_of_charge_set = stor_charge_end
								#p_nom = generators[gen_id]['Existing_Cap_MW'],
								#state_of_charge_initial = generators[gen_id]['Hydro_level'] * generators[gen_id]['Existing_Cap_MW'],
								
							)
			else:
				
				'Variability Added'
				try:
					variability_key = str(generators[gen_id]['Resource']) + '/' + str(generators[gen_id]['zone'])
					
					'RTM Variability perturbations'
					gen_var = variability[time_index][variability_key]
					if perturbate_variability:
						#perturbated_vars = variability_perturbations(generators[gen_id]['Resource'],variability[time_index][variability_key], network.snapshots, 1, gen_id, generators)
						perturbated_vars = Real_Time_Generator_Outage_Variability(generators[gen_id]['Resource'],
																				variability[time_index][variability_key], 
																				network.snapshots, 
																				1, 
																				gen_id, 
																				generators,cur_hour,cur_day)
						gen_var = perturbated_vars
					'Certain Generators need to scheduled on as they cannot be shutdown in RTM'
					if generators[gen_id]['Resource'] in Inflexible_Resources :
						
						if 'Nuclear' in generators[gen_id]['Resource']:
							network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom_extendable = True,
									p_nom_max = generators[gen_id]['Existing_Cap_MW'],
									p_nom_min = generators[gen_id]['Existing_Cap_MW'],
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									reserve_max = generators[gen_id]['Existing_Cap_MW'],
									p_max_pu = gen_var,
									p_min_pu = gen_var
									)
						else:
							network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom_extendable = True,
									p_nom_max = generators[gen_id]['Existing_Cap_MW'],
									p_nom_min = dam_coal_nuc_pow["{} {}".format(generators[gen_id]['Resource'], gen_id)][cur_hour], #Get thermal DAM dispatch
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									reserve_max = generators[gen_id]['Existing_Cap_MW'],
									p_max_pu = gen_var,
									p_min_pu = gen_var
									)
					else:
						network.add("Generator",
									"{} {}".format(generators[gen_id]['Resource'], gen_id),
									bus = generators[gen_id]['zone'],
									carrier = generators[gen_id]['Resource'],
									p_nom_extendable = True,
									p_nom_max = generators[gen_id]['Existing_Cap_MW'],
									p_nom_min = 0,
									marginal_cost = generators[gen_id]['Marginal_Cost'],
									reserve_max = generators[gen_id]['Existing_Cap_MW'],
									p_max_pu = gen_var
									)
				
				except KeyError:
					print('Error has occured in reading RTM variability.')
					network.add("Generator",
								"{} {}".format(generators[gen_id]['Resource'], gen_id),
								bus = generators[gen_id]['zone'],
								carrier = generators[gen_id]['Resource'],
								p_nom_extendable = True,
								p_nom_max = generators[gen_id]['Existing_Cap_MW'],
								p_nom_min = 0,
								marginal_cost = generators[gen_id]['Marginal_Cost'],
								reserve_max = generators[gen_id]['Existing_Cap_MW'],
								p_max_pu = 1.0
								)
		
		## V2G Participation in RTM
		if v2g_participation:
		
			v2g_bids = read_v2g_bids()
			
			#Currently only 1 day used
			day = 1
			for cur_bid in v2g_bids[day]:
			
				network.add("Generator",
							"{} {}".format('V2G ' + cur_bid["Bidder_ID"], cur_bid["Bid_ID"] ),
							bus = cur_bid['zone'],
							carrier = "V2G " + str(cur_bid["Bidder_ID"]),
							p_nom_extendable = True,
							marginal_cost = cur_bid["cost"],
							p_nom_max = cur_bid["capacity"],
							reserve_max = cur_bid["capacity"]*2,
				)
			
		## Add transmission lines
		for (zone0,zone1) in india_tranmission.keys():
		
			## Bi-directional and lossless links
			network.add("Link", 
						"{} - {}".format(zone0,zone1),
						bus0=zone0, bus1=zone1,
						marginal_cost=0.0,
						efficiency=1,
						p_nom=india_tranmission[(zone0,zone1)]['Line_Max_Flow_MW'],
						p_min_pu=-1,
						)
	else:
		'AGGREGATED'
		print('AGGREGATE RUN. WILL BREAK')
			
	return network


'''
Test Cases
'''

def india_scuc_case():

	global MIN_SYS_RESERVE_ENERGY_MW
	
	#MIN_SYS_RESERVE_ENERGY_MW = 2540.0
	MIN_SYS_RESERVE_ENERGY_MW = 0.0
	generators, cluster_generators = read_india_generators()
	india_tranmission ,zones = read_india_network()
	variability = read_india_gen_variability()
	
	## Initialize network
	network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
	
	
	is_aggregated = False
	#network = build_india_generators(network,generators,india_tranmission,zones,is_aggregated)
	network = ED_build_india_generators(network,generators,india_tranmission,zones,is_aggregated,variability,0)
	
	
	## Load per hour data
	loads, aggregated_loads, not_aggregated_loads = read_india_loads()
	

	## Set amount of snapshots to amount of Time_index
	#network.set_snapshots(range(len(loads)))
	network.set_snapshots(range(4))
	perturbated_15min_loads = demand_perturbations(not_aggregated_loads,is_aggregated,0,1,100)
	print(perturbated_15min_loads)
	network = add_india_loads(network,zones,is_aggregated,perturbated_15min_loads)
	'''
	if is_aggregated:
		network.add("Load", "{} load".format('India'), bus='India', p_set=(888952.08182  + 70329.73868 + 59710.81032 + 30750.63148 + 598.4472369)  )
		#network.add("Load", "{} load".format('India'), bus='India', p_set=aggregated_loads  )
	else:
		network.add("Load", "Zone {} load".format('1'), bus='1', p_set=888952.08182 )
		network.add("Load", "Zone {} load".format('2'), bus='2', p_set=70329.73868 )
		network.add("Load", "Zone {} load".format('3'), bus='3', p_set=59710.81032 )
		network.add("Load", "Zone {} load".format('4'), bus='4', p_set=30750.63148 )
		network.add("Load", "Zone {} load".format('5'), bus='5', p_set=598.4472369 )
	'''
	
	network.lopf(network.snapshots,extra_functionality=custom_MILP)
	#network.lopf()
	
	print(network.loads_t.p)
	print(network.generators_t.p)
	#print(calculate_curtailment(network))
	print(network.buses_t.marginal_price)

	
def india_DAM_RTM_algo(is_aggregated, time_index, days):
	'''
	India DAM and RTO/RTM model
	'''
	
	def get_demand_response_data(network):
		'''
		Returns demand response data from the network in dictionary format from Pyomo model
		'''
		try:
			demand_response_values = {}
			individual_demand_response_values = {}
		
			dr_keys = list(network.model.demand_response_dTotalDR.keys())
			dr_vals = list(network.model.demand_response_dTotalDR.values())
		
			ind_dr_keys = list(network.model.demand_response_dDR_bid.keys())
			ind_dr_vals = list(network.model.demand_response_dDR_bid.values())
		
			for i in range(len(dr_keys)) :
				demand_response_values[dr_keys[i]] = dr_vals[i].value
			
			for i in range(len(ind_dr_keys)):
				individual_demand_response_values[ind_dr_keys[i]] = ind_dr_vals[i].value
			
			return demand_response_values, individual_demand_response_values
		
		except:
			'No Demand Response Available'
			return None
	
	def get_rtm_demand_response_data(network):
		'''
		Returns the curtailed demand response in dictionary format from Pyomo
		'''
		try:
			demand_response_curtail_values = {}
			dr_curtail_keys = list(network.model.demand_response_dCurtailDR.keys())
			dr_curtail_vals = list(network.model.demand_response_dCurtailDR.values())
			
			for i in range(len(dr_curtail_keys)) :
				demand_response_curtail_values[dr_curtail_keys[i]] = dr_curtail_vals[i].value
			
			return demand_response_curtail_values

		except:
			'No RTM Demand Response Available'
			return None
	
	def get_reserve_data(network):
		
		reserve_values = {}
		r_keys = list(network.model.generator_r.keys())
		r_values = list(network.model.generator_r.values())
		
		r_stor_keys = list(network.model.storage_units_r.keys())
		r_stor_values = list(network.model.storage_units_r.values())
		
		
		#Extract generator reserves
		for i in range(len(r_keys)):
			reserve_values[r_keys[i]] = r_values[i].value
		#Extract storage reserves
		for i in range(len(r_stor_keys)):
			reserve_values[r_stor_keys[i]] = r_stor_values[i].value
			
		return reserve_values

	def get_unmet_data(network):
		'''
		Returns the unmet demand data in dictionary format from Pyomo model
		'''
		unmet_demand_values = {}
		unmet_reserve_values = {}
		
		#network.model.d_unmet.pprint()
		#network.model.r_unmet.pprint()
		
		d_unmet_keys = list(network.model.d_unmet.keys())
		d_unmet_values = list(network.model.d_unmet.values())
		
		r_unmet_keys = list(network.model.r_unmet.keys())
		r_unmet_values = list(network.model.r_unmet.values())
		
		
		for i in range(len(d_unmet_keys)):
			unmet_demand_values[d_unmet_keys[i]] = d_unmet_values[i].value
		for i in range(len(r_unmet_keys)):
			unmet_reserve_values[r_unmet_keys[i]] = r_unmet_values[i].value
			
		return unmet_demand_values, unmet_reserve_values
			
	global MIN_SYS_RESERVE_ENERGY_MW
	global CURRENT_TIME_INDEX
	global CURRENT_HOUR_INDEX
	global DAILY_FIXED_DR
	global GENERATOR_OUTAGE
	global GENERATOR_CLUSTER_OUTAGE
	
	'''RESET GLOBALS'''
	GENERATOR_OUTAGE = {}
	GENERATOR_CLUSTER_OUTAGE = []
	DAILY_FIXED_DR = None
	'''---'''
	
	generators, cluster_generators = read_india_generators()
	india_tranmission ,zones = read_india_network()
	loads, aggregated_loads, not_aggregated_loads = read_india_loads()
	variability = read_india_gen_variability()
	
	total_hours = len(loads)
	total_days = int(total_hours/24)
	
	previous_dam_gen_statuses = None
	previous_dam_storage_soc = None
	
	'String output variables'
	loads_rtm = ''
	generators_rtm = ''
	marginal_p_rtm = ''
	storage_p_rtm = ''
	storage_soc_rtm = ''
	
	loads_dam = ''
	generators_dam = ''
	generators_status_dam = ''
	storage_p_dam = ''
	storage_soc_dam = ''
	marginal_p_dam = ''
	
	rtm_networks = []
	rtm_curtailment = []
	rtm_vre_available = []
	rtm_dr_curtailment = []
	
	dam_uc_networks = []
	dam_ed_networks = []
	dam_dr_data = []
	dam_vre_available = []
	dam_curtailment = []
	
	dam_unmet_demand = []
	dam_unmet_reserve = []
	
	rtm_unmet_demand = []
	rtm_unmet_reserve = []
	
	dam_reserve_data = []
	
	outage_data = []
	
	
	start_time = time.time() #TIMING
	
	for day in range(total_days):
	
		CURRENT_TIME_INDEX = time_index
		
		'''Day Ahead Market'''
		## DAM Unit Commitment
		if is_aggregated:
			DAM_Loads = aggregated_loads[time_index: time_index + 24]
		else:
			DAM_Loads = {}
			DAM_Loads['1'] = not_aggregated_loads['1'][time_index: time_index + 24]
			DAM_Loads['2'] = not_aggregated_loads['2'][time_index: time_index + 24]
			DAM_Loads['3'] = not_aggregated_loads['3'][time_index: time_index + 24]
			DAM_Loads['4'] = not_aggregated_loads['4'][time_index: time_index + 24]
			DAM_Loads['5'] = not_aggregated_loads['5'][time_index: time_index + 24]
		
		# Currently min reserve is average of 24 hours
		MIN_SYS_RESERVE_ENERGY_MW = np.average(np.array(aggregated_loads[time_index: time_index + 24])) * 0.04 + 3000
		#MIN_SYS_RESERVE_ENERGY_MW = 0
		#print('Required Reserve: ' + str(MIN_SYS_RESERVE_ENERGY_MW) + '\n')
		
		## Security-Constrained Unit Commitment
		print('Day: '+str(int(time_index / 24) + 1)+' - Starting Security-Constrained Unit Commitment...\n')
		scuc_network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
		scuc_network.set_snapshots(range(24))
		#scuc_network = UC_build_india_generators(scuc_network,generators,india_tranmission,zones,is_aggregated,variability,time_index)
		#scuc_network = UC_build_india_generators(scuc_network,cluster_generators,india_tranmission,zones,is_aggregated,variability,time_index)
		
		scuc_network, VRE_Reserve = Day_Ahead_Unit_Commitment(scuc_network,cluster_generators,india_tranmission,zones,is_aggregated,variability,time_index,previous_dam_storage_soc)
		
		MIN_SYS_RESERVE_ENERGY_MW = MIN_SYS_RESERVE_ENERGY_MW + VRE_Reserve
		
		print('Required Reserve: ' + str(MIN_SYS_RESERVE_ENERGY_MW) + '\n')
		
		scuc_network = add_india_loads(scuc_network,zones,is_aggregated,DAM_Loads)
		scuc_network.lopf(extra_functionality=custom_MILP,solver_name=OPTIMIZATION_SOLVER)
		#with open('model.txt','w') as outfile:
		#	scuc_network.model.pprint(outfile)
		
		#For correct next day status
		previous_dam_storage_soc =  scuc_network.storage_units_t.state_of_charge
		
		
		## Find DAM LMPs
		print('Day: '+str(int(time_index / 24) + 1)+' - Finding DAM LMPs...\n')
		scuc_price_network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
		scuc_price_network.set_snapshots(range(24))
		#scuc_price_network = Fixed_ED_build_india_generators(scuc_price_network,generators,india_tranmission,zones,is_aggregated,variability,time_index,scuc_network.generators_t.status)
		scuc_price_network = Fixed_ED_build_india_generators(scuc_price_network,cluster_generators,india_tranmission,zones,is_aggregated,variability,time_index,scuc_network.generators_t.status)
		scuc_price_network = add_india_loads(scuc_price_network,zones,is_aggregated,DAM_Loads)
		scuc_price_network.lopf(extra_functionality=custom_MILP,solver_name=OPTIMIZATION_SOLVER)
		
		#Extract Curtailment Values
		curtailment, vre_available = calculate_curtailment(scuc_network)
		
		dam_vre_available.append(vre_available)
		dam_curtailment.append(curtailment)
		dam_uc_networks.append(scuc_network)
		dam_ed_networks.append(scuc_price_network)
		outage_data.append(copy.deepcopy(GENERATOR_CLUSTER_OUTAGE))
		
		#Extract Unmet Penalty values
		unmet_demand_values, unmet_reserve_values = get_unmet_data(scuc_network)
		
		dam_unmet_demand.append(unmet_demand_values)
		dam_unmet_reserve.append(unmet_reserve_values)
		
		#Extract reserve values
		reserve_data = get_reserve_data(scuc_network)
		dam_reserve_data.append(reserve_data)
		
		'Add Demand response to DAM Loads'
		try: 
			
			dr_data, ind_dr_data = get_demand_response_data(scuc_network)
			dam_dr_data.append(dr_data)
			DAILY_FIXED_DR = dr_data
			for (zone, snapshot) in dr_data.keys():
				if is_aggregated:
					DAM_Loads[snapshot] = DAM_Loads[snapshot] + dr_data[(zone,snapshot)]
				else:
					DAM_Loads[zone][snapshot] = DAM_Loads[zone][snapshot] + dr_data[(zone,snapshot)]
		except:
			print('No Demand Response')
		
		'Sim Outputs'
		loads_dam = loads_dam + scuc_network.loads_t.p.to_csv(index_label='snapshots',line_terminator='\n')
		generators_dam = generators_dam + scuc_network.generators_t.p.to_csv(index_label='snapshots',line_terminator='\n')
		storage_p_dam = storage_p_dam + scuc_network.storage_units_t.p.to_csv(index_label='snapshots',line_terminator='\n')
		storage_soc_dam = storage_soc_dam + scuc_network.storage_units_t.state_of_charge.to_csv(index_label='snapshots',line_terminator='\n')
		
		generators_status_dam = generators_status_dam + scuc_network.generators_t.status.to_csv(index_label='snapshots',line_terminator='\n')
		marginal_p_dam = marginal_p_dam + scuc_price_network.buses_t.marginal_price.to_csv(index_label='snapshots',line_terminator='\n')
		
		'''Real Time Market'''
		## RTM Economic Dispatch
		print('Running RTM...')
		hours_in_day = range(time_index, time_index + 24)
		
		'Track the scheduled power of DAM for RTM'
		dam_storage_units = scuc_network.storage_units_t
		dam_stor_pow = {}
		dam_stor_soc = {}
		for (name, stor_power) in dam_storage_units.p.iteritems():
			dam_stor_pow[name] = stor_power
		for (name, stor_soc) in dam_storage_units.state_of_charge.iteritems():
			dam_stor_soc[name] = stor_soc
		
		dam_generator_units = scuc_network.generators_t
		dam_coal_nuc_pow = {}
		for (name, gen_pow) in dam_generator_units.p.iteritems():
			if 'Coal' in name or 'Nuclear' in name:
				dam_coal_nuc_pow[name] = gen_pow
		
		for hour in hours_in_day:
			
			CURRENT_HOUR_INDEX = hour % 24
			
			
			
			lped_network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
			lped_network.set_snapshots(range(4))
			
			#lped_network = ED_build_india_generators(lped_network,cluster_generators,india_tranmission,zones,is_aggregated,variability,time_index,True)
			lped_network = Real_Time_Economic_Dispatch(lped_network,cluster_generators,india_tranmission,zones,is_aggregated,variability,time_index,True,True,CURRENT_HOUR_INDEX,day, dam_stor_pow, dam_stor_soc, dam_coal_nuc_pow)
			
			# Set the RTM Loads
			if is_aggregated:
				RTM_Loads = DAM_Loads[time_index % 24: time_index % 24 + 2] 
			else:
				RTM_Loads = {}
				RTM_Loads['1'] = DAM_Loads['1'][time_index % 24 : time_index % 24 + 2]
				RTM_Loads['2'] = DAM_Loads['2'][time_index % 24 : time_index % 24 + 2]
				RTM_Loads['3'] = DAM_Loads['3'][time_index % 24 : time_index % 24 + 2]
				RTM_Loads['4'] = DAM_Loads['4'][time_index % 24 : time_index % 24 + 2]
				RTM_Loads['5'] = DAM_Loads['5'][time_index % 24 : time_index % 24 + 2]
				
			
			# Use perturbated loads for each 15 min
			perturbated_15min_loads = demand_perturbations(RTM_Loads,is_aggregated,0,1,5)
			identical_loads = {}
			identical_loads['1'] = np.array(4*[RTM_Loads['1'][0]])
			identical_loads['2'] = np.array(4*[RTM_Loads['2'][0]])
			identical_loads['3'] = np.array(4*[RTM_Loads['3'][0]])
			identical_loads['4'] = np.array(4*[RTM_Loads['4'][0]])
			identical_loads['5'] = np.array(4*[RTM_Loads['5'][0]])
			
			lped_network = add_india_loads(lped_network,zones,is_aggregated,identical_loads)
			lped_network.lopf(extra_functionality=custom_MILP,solver_name=OPTIMIZATION_SOLVER)
			#with open('model.txt','w') as outfile:
			#	lped_network.model.pprint(outfile)

			#Extract Unmet Penalty values
			unmet_demand_values, unmet_reserve_values = get_unmet_data(lped_network)
			rtm_unmet_demand.append(unmet_demand_values)
			rtm_unmet_reserve.append(unmet_reserve_values)
			
			#Extract Curtailment
			print()
			print('HOUR: ' + str(hour))
			curtailment_rtm, vre_available_rtm = calculate_curtailment(lped_network)
			rtm_vre_available.append(vre_available_rtm)
			rtm_curtailment.append(curtailment_rtm)
			
			#Extract RTM DR Values
			try:
				demand_response_curtail_values = get_rtm_demand_response_data(lped_network)
				rtm_dr_curtailment.append(demand_response_curtail_values)
			except:
				print('No RTM DR')
			
			
			loads_rtm = loads_rtm + lped_network.loads_t.p.to_csv(index_label='snapshots',line_terminator='\n')
			generators_rtm = generators_rtm + lped_network.generators_t.p.to_csv(index_label='snapshots',line_terminator='\n')
			marginal_p_rtm = marginal_p_rtm + lped_network.buses_t.marginal_price.to_csv(index_label='snapshots',line_terminator='\n')
			storage_p_rtm = storage_p_rtm + lped_network.storage_units_t.p.to_csv(index_label='snapshots',line_terminator='\n')
			storage_soc_rtm = storage_soc_rtm + lped_network.storage_units_t.state_of_charge.to_csv(index_label='snapshots',line_terminator='\n')
			
			rtm_networks.append(lped_network)
			
			time_index = time_index + 1
			
		if day >= (days - 1):
			break
	'''
	Writing Outputs
	
	write_output('dam_loads.csv',loads_dam)
	write_output('dam_generator_power.csv',generators_dam)
	write_output('dam_generator_uc.csv',generators_status_dam)

	write_output('rtm_loads.csv',loads_rtm)
	write_output('rtm_generator_power.csv',generators_rtm)
	write_output('rtm_storage_power.csv',storage_p_rtm)
	write_output('rtm_storage_soc.csv',storage_soc_rtm)
	write_output('rtm_prices.csv',marginal_p_rtm)
	
	write_output('uc_fixed_generator_power.csv',scuc_price_network.generators_t.p.to_csv(index_label='snapshots',line_terminator='\n'))
	write_output('dam_prices.csv',marginal_p_dam)

	write_tableau_output('tableau_output\\',rtm_networks,dam_uc_networks,dam_ed_networks,
						 generators,cluster_generators,
						 dam_dr_data,outage_data, 
						 STARTING_TIME_BLOCK + 1, SCENARIO_NAME, 'w' )
	'''
	
	print("ELASPED TIME: " + str(time.time() - start_time) + ' seconds for Time Block: ' + str(STARTING_TIME_BLOCK + 1) + ' with ' + str(days) + ' days.' )
	
	return ([loads_dam, generators_dam, generators_status_dam,  marginal_p_dam, loads_rtm, generators_rtm, storage_p_rtm, storage_soc_rtm, marginal_p_rtm, storage_p_dam, storage_soc_dam],
		   [rtm_networks, dam_uc_networks, dam_ed_networks, generators, cluster_generators, dam_dr_data, outage_data, dam_vre_available, dam_curtailment, dam_unmet_demand, dam_unmet_reserve, 
		   rtm_unmet_demand, rtm_unmet_reserve, rtm_vre_available, rtm_curtailment, rtm_dr_curtailment, dam_reserve_data])
	
	
def test1():
	
	#marginal costs in EUR/MWh
	marginal_costs = {"Wind" : 0,
				"Hydro" : 0,
				"Coal" : 30,
				"Gas" : 60,
				"Oil" : 80}
				
	#power plant capacities (nominal powers in MW) in each country (not necessarily realistic)
	power_plant_p_nom = {"South Africa" : {"Coal" : 47000, "Wind" : 7000,"Gas" : 8000, "Oil" : 2000},
							"Mozambique" : {"Hydro" : 1200,},
							"Swaziland" : {"Hydro" : 600,},
						}
						
	#transmission capacities in MW (not necessarily realistic)
	transmission = {"South Africa" : {"Mozambique" : 10000,
						"Swaziland" : 6500},
						"Mozambique" : {"Swaziland" : 2500}}

	#country electrical loads in MW (not necessarily realistic)
	loads = {"South Africa" : 5000,
			"Mozambique" : 6500,
			"Swaziland" : 2500}
	
	#Designing Network T=0,1,2,3
	network = pypsa.Network()
	network.set_snapshots(range(4))
	countries = ["South Africa", "Mozambique", "Swaziland"]
	
	for country in countries:
		
		network.add("Bus",country)
		
		for tech in power_plant_p_nom[country]:
			network.add("Generator",
						"{} {}".format(country,tech),
						bus=country,
						p_nom=power_plant_p_nom[country][tech],
						marginal_cost = marginal_costs[tech],
						p_max_pu=([0.3,0.6,0.4,0.5] if tech == "Wind" else 1),
			)
		
		network.add("Load", "{} load".format(country), bus=country, p_set=loads[country] + np.array([0,1000,3000,4000]) )
		
		#add transmission as controllable Link
		if country not in transmission:
			continue
		
		for other_country in countries:
		
			if other_country not in transmission[country]:
				continue
				
			#NB: Link is by default unidirectional, so have to set p_min_pu = -1
			#to allow bidirectional (i.e. also negative) flow

			network.add("Link", 
						"{} - {}".format(country,other_country),
						bus0=country, bus1=other_country,
						p_nom=transmission[country][other_country],
						p_min_pu=-1)
	network.lopf()
	print(network.loads_t.p)
	print(network.generators_t.p)
	print(network.storage_units_t.p)
	print(network.storage_units_t.state_of_charge)
	print(network.buses_t.marginal_price)
	
	n.loads_t.p_set.sum(axis=1).max()
	
	fig,ax = plt.subplots(1,1,subplot_kw={"projection":ccrs.PlateCarree()})
	fig.set_size_inches(6,6)
	load_distribution = network.loads_t.p_set.loc[network.snapshots[0]].groupby(network.loads.bus).sum()
	print(load_distribution)
	network.plot(bus_sizes=0.5*load_distribution,ax=ax,title="Load distribution")
	fig.savefig("load.png")

def test2():

	output_file("output1.html")
	
	generators = {}
	prod_pw = []
	prod_price = []
	
	consumers = {}
	cons_pw = []
	cons_price = []
	
	with open('india_data_cases\\textbook_grid_data\\supply_stack.csv','r') as supplyfile:
		supplyfile.readline()
		for line in supplyfile.readlines():
			params = line.split(',')
			generators[params[0]] = {}
			generators[params[0]]['Type'] = params[1]
			generators[params[0]]['Price'] = float(params[2].replace('$','').replace(' ',''))
			generators[params[0]]['Power'] = float(params[3].replace('\n',''))
			
			prod_price.append(params[2].replace('$','').replace(' ',''))
			if(len(prod_pw) == 0):
				prod_pw.append(float(params[3].replace('\n','')))
			else:
				prod_pw.append(float(params[3].replace('\n',''))+ prod_pw[len(prod_pw) - 1])
	
	with open('india_data_cases\\textbook_grid_data\\demand_stack.csv','r') as supplyfile:
		supplyfile.readline()
		for line in supplyfile.readlines():
			params = line.split(',')
			consumers[params[0]] = {}
			consumers[params[0]]['Price'] = float(params[1].replace('$','').replace(' ',''))
			consumers[params[0]]['Power'] = float(params[2].replace('\n',''))
			
			cons_price.append(params[1].replace('$','').replace(' ',''))
			if(len(cons_pw) == 0):
				cons_pw.append(float(params[2].replace('\n','')))
			else:
				cons_pw.append(float(params[2].replace('\n','')) + cons_pw[len(cons_pw) - 1])
				
	#'Plot Supply Demand Curves'
	p = figure(plot_width=800, plot_height=800)
	p.step(prod_pw, prod_price, line_width=2, mode="center",color='Green')
	p.step(cons_pw, cons_price, line_width=2, mode="center",color='Red')
	show(p)
	
	#'Start Network'
	network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
	network.set_snapshots(range(2))
	
	#'Load Parameters'
	regions = ['California']
	marginal_costs = {}
	power_plant_p_nom = {}
	
	for region in regions:
		power_plant_p_nom[region] = {}
		
		for gen_name in generators.keys():
			marginal_costs[gen_name] = generators[gen_name]['Price']
			power_plant_p_nom[region][gen_name] = generators[gen_name]['Power']
		
		network.add("Bus",region)
		
		for gen in power_plant_p_nom[region]:
			network.add("Generator",
						"{} {}".format(region,gen),
						bus=region,
						p_nom_max=power_plant_p_nom[region][gen],
						p_nom_min=0,
						marginal_cost = marginal_costs[gen],
						p_max_pu=1,
						carrier=generators[gen]['Type'],
						reserve_max = 100,
						p_nom_extendable=True
			)
			
		network.add("Load", "{} load".format(region), bus=region, p_set=cons_pw[len(cons_pw) - 1] )
	
	#'Clear the Market'
	network.lopf(extra_functionality=reserve_constraints)
	#network.lopf()
	
	print('\n')
	print("Marginal Price:")
	print(network.buses_t.marginal_price)
	print(network.buses_t.marginal_price)
	print("Total revenue:")
	print(network.generators_t.p.multiply(network.snapshot_weightings,axis=0).multiply(network.buses_t.marginal_price["California"],axis=0).sum())
	
	print('\n')
	print(cons_pw[len(cons_pw) - 1])
	print(prod_pw[len(prod_pw) - 1])
	print(power_plant_p_nom)
	

def textbook_example():

	global MIN_SYS_RESERVE_ENERGY_MW
	MIN_SYS_RESERVE_ENERGY_MW = 250.0
	network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
	network.set_snapshots(range(4))
	
	network.add("Bus","A")
	network.add("Generator",
				"{} {}".format('A','1'),
				bus='A',
				p_nom_max=250.0,
				p_nom_min=0,
				marginal_cost = 2.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','2'),
				bus='A',
				p_nom_max=230.0,
				p_nom_min=0,
				marginal_cost = 17.0,
				p_max_pu=1,
				reserve_max = 160,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','3'),
				bus='A',
				p_nom_max=240.0,
				p_nom_min=0,
				marginal_cost = 20.0,
				p_max_pu=1,
				reserve_max = 190,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','4'),
				bus='A',
				p_nom_max=250,
				p_nom_min=0,
				marginal_cost = 28.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)
	network.add("Load", "{} load".format('A'), bus='A', p_set=[400,475,725,730])
	
	network.lopf(extra_functionality=custom_MILP)

	#output_file("textbook.html")
	#p = figure(plot_width=800, plot_height=800)
	#p.step([250,480,720,970], [2,17,20,28], line_width=2, mode="center",color='Green')
	#show(p)
	
	print(network.model.pprint())
	print(network.buses_t.marginal_price)
	print(network.generators_t.p)

def DR_test():

	global MIN_SYS_RESERVE_ENERGY_MW
	MIN_SYS_RESERVE_ENERGY_MW = 250.0
	network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
	network.set_snapshots(range(24))
	
	network.add("Bus",'1')
	network.add("Bus",'2')
	'''
	network.add("Generator",
				"{} {}".format('A','Solar'),
				bus='A',
				p_nom_max=2500.0,
				p_nom_min=2500.0,
				#marginal_cost = 0.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)'''
	network.add("Generator",
				"{} {}".format('A','1'),
				bus='1',
				p_nom_max=2500.0,
				p_nom_min=0,
				marginal_cost = 2.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','2'),
				bus='1',
				p_nom_max=2300.0,
				p_nom_min=0,
				marginal_cost = 17.0,
				p_max_pu=1,
				reserve_max = 160,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','3'),
				bus='1',
				p_nom_max=2400.0,
				p_nom_min=0,
				marginal_cost = 20.0,
				p_max_pu=1,
				reserve_max = 190,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','4'),
				bus='1',
				p_nom_max=2500,
				p_nom_min=0,
				marginal_cost = 28.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)
	loads = [475,475,4750,4750,
			4750,4750,4750,4750,
			4750,4750,475,4750,
			4750,475,4750,4750,
			4750,4750,4750,4750,
			4750,4750,4750,4750]
	loads2 = [475,475,4750,4750,
			475,475,475,475,
			4750,4750,475,475,
			4750,475,4750,475,
			4750,4750,4750,4750,
			4750,4750,4750,4750]
	network.add("Load", "{} load".format('1'), bus='1', p_set=loads)
	network.add("Load", "{} load".format('2'), bus='2', p_set=loads2)
	network.add("Link", 
				"{} - {}".format('1','2'),
				bus0='1', bus1='2',
				marginal_cost=0.0,
				efficiency=1,
				p_nom=20000.0,
				p_min_pu=-1)
				
	network.lopf(extra_functionality=custom_MILP)
	#network.lopf()
	
	print(network.model.pprint())
	print(network.buses_t.marginal_price)
	print(network.generators_t.p)
	print(network.loads_t.p)
	
	demand_response_values = {}
	
	dr_keys = list(network.model.demand_response_dTotalDR.keys())
	dr_vals = list(network.model.demand_response_dTotalDR.values())
	for i in range(len(dr_keys)) :
		demand_response_values[dr_keys[i]] = dr_vals[i].value
		
	print(demand_response_values)
	
def DR_test_aggregated():
	global MIN_SYS_RESERVE_ENERGY_MW
	MIN_SYS_RESERVE_ENERGY_MW = 250.0
	network = pypsa.Network(override_components=override_components, override_component_attrs=override_component_attrs)
	network.set_snapshots(range(24))
	
	network.add("Bus",'A')
	'''
	network.add("Generator",
				"{} {}".format('A','Solar'),
				bus='A',
				p_nom_max=2500.0,
				p_nom_min=2500.0,
				#marginal_cost = 0.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)'''
	network.add("Generator",
				"{} {}".format('A','1'),
				bus='A',
				p_nom_max=2500.0,
				p_nom_min=0,
				marginal_cost = 2.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','2'),
				bus='A',
				p_nom_max=2300.0,
				p_nom_min=0,
				marginal_cost = 17.0,
				p_max_pu=1,
				reserve_max = 160,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','3'),
				bus='A',
				p_nom_max=2400.0,
				p_nom_min=0,
				marginal_cost = 20.0,
				p_max_pu=1,
				reserve_max = 190,
				p_nom_extendable=True
			)
	network.add("Generator",
				"{} {}".format('A','4'),
				bus='A',
				p_nom_max=2500,
				p_nom_min=0,
				marginal_cost = 28.0,
				p_max_pu=1,
				reserve_max = 0,
				p_nom_extendable=True
			)
	loads = [475,475,4750,4750,
			4750,4750,4750,4750,
			4750,4750,475,4750,
			4750,475,4750,4750,
			4750,4750,4750,4750,
			4750,4750,4750,4750]
	network.add("Load", "{} load".format('A'), bus='A', p_set=loads)
	
	network.lopf(extra_functionality=custom_MILP)
	#network.lopf()
	
	print(network.model.pprint())
	print(network.buses_t.marginal_price)
	print(network.generators_t.p)
	
	print(network.loads_t.p)

def run_scenarios():
	
	'''Global Parameters'''
	global INDIA_DATA_DIR		#INPUT Directory
	global SCENARIO_NAME		#i.e. L,H,8,1275
	global SIM_OUTPUT_DIR		#sim_output Directory - pypsa outputs
	global TABLEAU_OUTPUT_DIR	#tableau_output Directory - outputs formatted for tableau
	global DEMAND_RESPONSE_BIDS	#Demand response bids filename
	global DEFAULT_NO_CHARGING	#Boolean for default case (set true to run default)
	global BIDS_FOLDER			#Folder containing all the demand response bids
	global SCENARIO_NAME		#Scenario Name
	global ALL_SCENARIOS_OUTPUT_FOLDER	#Current scenario output folder (above sim_output and tableau_output)
	global STARTING_TIME_BLOCK	#The current timeblock [0,1,2,3]
	global GENERATOR_CLUSTER_SIZE #i.e. 4 Captures 25%, 50%, 75%, or 100% of fleet on or off
	global OPTIMIZATION_SOLVER
	global V2G_BIDS
	
	ALL_SCENARIOS_OUTPUT_FOLDER = "simulation_outputs\\"
	
	SCENARIO_NAME = 'L,L,11,1275'
	INDIA_DATA_DIR = 'india_data_cases\\tony-2\\'
	INDIA_DATA_DIR = INDIA_DATA_DIR + SCENARIO_NAME + '\\'
	
	
	#scenarios = ['default']
	scenarios = ['real-med-morning']
	'DEFAULT_NO_CHARGING = True #PLEASE SET THIS TRUE WHEN RUNNING DEFAULT. Otherwise will try to read DR Bids'
	'Setting scenario to default will set this to true'
	#scenarios = ['default','real-low-morning','real-med-morning','real-high-morning', 'real-low-evening', 'real-med-evening', 'real-high-evening']
	
	GENERATOR_CLUSTER_SIZE = 1
	OPTIMIZATION_SOLVER = 'cbc'
	
	#time_block_starting_indexes = [2641,3961,6841,8281]
	time_block_starting_indexes = [2641,3961,6841,8281]
	time_index = time_block_starting_indexes[STARTING_TIME_BLOCK] 
	
	real_date = datetime.datetime.now().strftime("%m-%d-%Y") # For the output folder
	
	days = 5
	
	'String output variables'
	loads_dam_all = ''
	generators_dam_all = ''
	generators_status_dam_all = ''
	storage_p_dam_all = ''
	storage_soc_dam_all = ''
	marginal_p_dam_all = ''
	
	loads_rtm_all = ''
	generators_rtm_all = ''
	storage_p_rtm_all = ''
	storage_soc_rtm_all = ''
	marginal_p_rtm_all = ''
	
	rtm_networks_all = []
	
	
	'''Runs Every Demand Response Scenario'''
	for scenario_name in scenarios:
	
		'Ensures that EV Bids will not be read when running default'
		if scenario_name == 'default':
			DEFAULT_NO_CHARGING = True
		else:
			DEFAULT_NO_CHARGING = False
			
		DEMAND_RESPONSE_BIDS = BIDS_FOLDER + scenario_name + '-bids.csv'
		
		scenario_folder = ALL_SCENARIOS_OUTPUT_FOLDER + scenario_name + '-' + real_date +'-' + SCENARIO_NAME +'//'
		scenario_folder_DR = scenario_folder + 'demand-response//'
		try:
			os.makedirs(scenario_folder_DR + SIM_OUTPUT_DIR)
		except:
			print(SIM_OUTPUT_DIR + ' Directory already exists')
		try:
			os.makedirs(scenario_folder_DR + TABLEAU_OUTPUT_DIR)
		except:
			print(TABLEAU_OUTPUT_DIR + ' Directory already exists')
		
		
		'''Runs Every Timeblock'''
		for start_time_index in range(len(time_block_starting_indexes)):
	
			STARTING_TIME_BLOCK = start_time_index
			time_index = time_block_starting_indexes[STARTING_TIME_BLOCK]
		
			'''RUNNING SIMULATION'''
			sim_outputs, tableau_outputs = india_DAM_RTM_algo(False,time_index,days)
			
			'OUTPUTS'
			'[loads_dam, generators_dam, generators_status_dam,  marginal_p_dam, loads_rtm, generators_rtm, storage_p_rtm, storage_soc_rtm, marginal_p_rtm]'
			'[rtm_networks, dam_uc_networks, dam_ed_networks, generators, cluster_generators, dam_dr_data, outage_data]'
			'''---'''
			
			rtm_networks = tableau_outputs[0]
			dam_uc_networks = tableau_outputs[1]
			dam_ed_networks = tableau_outputs[2]
			generators = tableau_outputs[3]
			cluster_generators = tableau_outputs[4]
			dam_dr_data = tableau_outputs[5]
			outage_data = tableau_outputs[6]
			dam_vre_available = tableau_outputs[7]
			dam_curtailment = tableau_outputs[8]
			dam_unmet_demand = tableau_outputs[9]
			dam_unmet_reserve = tableau_outputs[10]
			rtm_unmet_demand = tableau_outputs[11]
			rtm_unmet_reserve = tableau_outputs[12]
			rtm_vre_available = tableau_outputs[13]
			rtm_curtailment = tableau_outputs[14]
			rtm_dr_curtailment = tableau_outputs[15]
			dam_reserve_data = tableau_outputs[16]
			
			
			loads_dam_all = loads_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[0]
			generators_dam_all = generators_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[1]
			generators_status_dam_all = generators_status_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[2]
			marginal_p_dam_all = marginal_p_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[3]
		
			loads_rtm_all = loads_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[4]
			generators_rtm_all = generators_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[5]
			storage_p_rtm_all = storage_p_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[6]
			storage_soc_rtm_all = storage_soc_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[7]
			marginal_p_rtm_all = marginal_p_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[8]
			
			storage_p_dam_all = storage_p_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[9]
			storage_soc_dam_all = storage_soc_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[10]
			
			if start_time_index == 0:
				write_tableau_output(scenario_folder_DR + TABLEAU_OUTPUT_DIR,rtm_networks,dam_uc_networks,dam_ed_networks,
									generators,cluster_generators,
									dam_dr_data,outage_data, dam_vre_available, dam_curtailment, dam_unmet_demand, dam_unmet_reserve,
									rtm_unmet_demand, rtm_unmet_reserve, rtm_vre_available, rtm_curtailment, rtm_dr_curtailment, dam_reserve_data,
									STARTING_TIME_BLOCK + 1, scenario_name +'-'+ SCENARIO_NAME.replace(',','-')  + '-flexible', 'w' )
			else:
				write_tableau_output(scenario_folder_DR + TABLEAU_OUTPUT_DIR,rtm_networks,dam_uc_networks,dam_ed_networks,
									generators,cluster_generators,
									dam_dr_data,outage_data, dam_vre_available, dam_curtailment, dam_unmet_demand, dam_unmet_reserve,
									rtm_unmet_demand, rtm_unmet_reserve, rtm_vre_available, rtm_curtailment, rtm_dr_curtailment, dam_reserve_data,
									STARTING_TIME_BLOCK + 1, scenario_name +'-'+ SCENARIO_NAME.replace(',','-')  + '-flexible', 'a' )
			
		'''Writing Aggregated Outputs'''
		write_output('dam_loads.csv',loads_dam_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('dam_generator_power.csv',generators_dam_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('dam_generator_uc.csv',generators_status_dam_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('dam_prices.csv',marginal_p_dam_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('dam_storage_p.csv',storage_p_dam_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('dam_storage_soc.csv',storage_soc_dam_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		
		write_output('rtm_loads.csv',loads_rtm_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('rtm_generator_power.csv',generators_rtm_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('rtm_storage_power.csv',storage_p_rtm_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('rtm_storage_soc.csv',storage_soc_rtm_all,scenario_folder_DR + SIM_OUTPUT_DIR)
		write_output('rtm_prices.csv',marginal_p_rtm_all,scenario_folder_DR + SIM_OUTPUT_DIR)
	
	
	
	'String output variables'
	loads_dam_all = ''
	generators_dam_all = ''
	generators_status_dam_all = ''
	storage_p_dam_all = ''
	storage_soc_dam_all = ''
	marginal_p_dam_all = ''
	
	loads_rtm_all = ''
	generators_rtm_all = ''
	storage_p_rtm_all = ''
	storage_soc_rtm_all = ''
	marginal_p_rtm_all = ''
	
	rtm_networks_all = []
	'''Runs Every NO Demand Response Scenario'''
	for scenario_name in scenarios:
		if not scenario_name == 'default':
			DEFAULT_NO_CHARGING = False
			
			DEMAND_RESPONSE_BIDS = BIDS_FOLDER + 'No_DR_' + scenario_name + '-bids.csv'
		
			scenario_folder = ALL_SCENARIOS_OUTPUT_FOLDER + scenario_name + '-' + real_date + '-' + SCENARIO_NAME + '//'
			scenario_folder_No_DR = scenario_folder + 'no-demand-response//'
			try:
				os.makedirs(scenario_folder_No_DR + SIM_OUTPUT_DIR)
			except:
				print(SIM_OUTPUT_DIR + ' Directory already exists')
			try:
				os.makedirs(scenario_folder_No_DR + TABLEAU_OUTPUT_DIR)
			except:
				print(TABLEAU_OUTPUT_DIR + ' Directory already exists')
		
		
			'''Runs Every Timeblock'''
			for start_time_index in range(len(time_block_starting_indexes)):
	
				STARTING_TIME_BLOCK = start_time_index
				time_index = time_block_starting_indexes[STARTING_TIME_BLOCK]
		
				'''RUNNING SIMULATION'''
				sim_outputs, tableau_outputs = india_DAM_RTM_algo(False,time_index,days)
				#OUTPUTS
				#[loads_dam, generators_dam, generators_status_dam,  marginal_p_dam, loads_rtm, generators_rtm, storage_p_rtm, storage_soc_rtm, marginal_p_rtm]
				#[rtm_networks, dam_uc_networks, dam_ed_networks, generators, cluster_generators, dam_dr_data, outage_data]
				'''---'''
			
				rtm_networks = tableau_outputs[0]
				dam_uc_networks = tableau_outputs[1]
				dam_ed_networks = tableau_outputs[2]
				generators = tableau_outputs[3]
				cluster_generators = tableau_outputs[4]
				dam_dr_data = tableau_outputs[5]
				outage_data = tableau_outputs[6]
				dam_vre_available = tableau_outputs[7]
				dam_curtailment = tableau_outputs[8]
				dam_unmet_demand = tableau_outputs[9]
				dam_unmet_reserve = tableau_outputs[10]
				rtm_unmet_demand = tableau_outputs[11]
				rtm_unmet_reserve = tableau_outputs[12]
				rtm_vre_available = tableau_outputs[13]
				rtm_curtailment = tableau_outputs[14]
				rtm_dr_curtailment = tableau_outputs[15]
				dam_reserve_data = tableau_outputs[16]
				
				loads_dam_all = loads_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[0]
				generators_dam_all = generators_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[1]
				generators_status_dam_all = generators_status_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[2]
				marginal_p_dam_all = marginal_p_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[3]
		
				loads_rtm_all = loads_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[4]
				generators_rtm_all = generators_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[5]
				storage_p_rtm_all = storage_p_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[6]
				storage_soc_rtm_all = storage_soc_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[7]
				marginal_p_rtm_all = marginal_p_rtm_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[8]
			
				storage_p_dam_all = storage_p_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[9]
				storage_soc_dam_all = storage_soc_dam_all + 'TIMEBLOCK: ' + str(STARTING_TIME_BLOCK + 1) + '\n' + sim_outputs[10]
			
				if start_time_index == 0:
					write_tableau_output(scenario_folder_No_DR + TABLEAU_OUTPUT_DIR,rtm_networks,dam_uc_networks,dam_ed_networks,
										generators,cluster_generators,
										dam_dr_data,outage_data, dam_vre_available, dam_curtailment, dam_unmet_demand, dam_unmet_reserve, rtm_unmet_demand, rtm_unmet_reserve, 
										rtm_vre_available, rtm_curtailment, rtm_dr_curtailment, dam_reserve_data,
										STARTING_TIME_BLOCK + 1, scenario_name +'-'+ SCENARIO_NAME.replace(',','-') + '-inflexible', 'w' )
				else:
					write_tableau_output(scenario_folder_No_DR + TABLEAU_OUTPUT_DIR,rtm_networks,dam_uc_networks,dam_ed_networks,
										generators,cluster_generators,
										dam_dr_data,outage_data, dam_vre_available, dam_curtailment, dam_unmet_demand, dam_unmet_reserve, 
										rtm_unmet_demand, rtm_unmet_reserve, rtm_vre_available, rtm_curtailment, rtm_dr_curtailment, dam_reserve_data,
										STARTING_TIME_BLOCK + 1, scenario_name +'-'+ SCENARIO_NAME.replace(',','-') + '-inflexible', 'a' )
			
			'''Writing Aggregated Outputs'''
			write_output('dam_loads.csv',loads_dam_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('dam_generator_power.csv',generators_dam_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('dam_generator_uc.csv',generators_status_dam_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('dam_prices.csv',marginal_p_dam_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('dam_storage_p.csv',storage_p_dam_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('dam_storage_soc.csv',storage_soc_dam_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			
			write_output('rtm_loads.csv',loads_rtm_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('rtm_generator_power.csv',generators_rtm_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('rtm_storage_power.csv',storage_p_rtm_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('rtm_storage_soc.csv',storage_soc_rtm_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			write_output('rtm_prices.csv',marginal_p_rtm_all,scenario_folder_No_DR + SIM_OUTPUT_DIR)
			
			

if __name__ == '__main__':
	add_reserve_attributes()
	
	#textbook_example()
	#all_EV_bids = read_EV_demand_response_bids()
	#print(all_EV_bids[1])
	#print(len(all_EV_bids[1][0]['charging_hours_min']))
	#DR_test()
	#DR_test_aggregated()
	#loads, aggregated_loads, not_aggregated_loads = read_india_loads()
	#print(demand_perturbations(aggregated_loads,True,0,1))
	#india_scuc_case()
	
	'Run Algo'
	#name = read_india_generators()
	run_scenarios()
	
	print(GENERATOR_OUTAGE)
	print(GENERATOR_CLUSTER_OUTAGE)




