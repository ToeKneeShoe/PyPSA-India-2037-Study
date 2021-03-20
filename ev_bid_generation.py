'''
Tony Shu
Electricity Market - Bid Generation
'''

import numpy as np
import pandas as pd #Only v.23 works as .24 removes sort in append
import os
import random
import matplotlib.pyplot as plt
import scipy.special
import copy

import json

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file


'''
GLOBAL VARIABLES
'''

OUTPUT_FILE_NAME = 'GENERATED_EV_bids_data.csv'
INDIA_DATA_DIR = 'india_data_cases\\Generated_EV_Data\\'
AGGREGATOR_DATA_FOLDER = 'aggregator_data\\'
SCENARIO_FOLDER = 'aggregator_data\\Scenarios\\'
SCENARIO_OUTPUT_FOLDER = 'aggregator_data\\Generated_Data\\'
BID_ID_INDEX = 0
BIDDER_IDS = ['Tony','Mike','Dimitri','Uber']

'''
Utility Functions
'''
def make_plot(title, hist, edges, x, pdf, cdf):
	'''From https://bokeh.pydata.org/en/latest/docs/gallery/histogram.html '''
	p = figure(title=title, tools='', background_fill_color="#fafafa")
	if hist is not None:
		p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
		fill_color="navy", line_color="white", alpha=0.5)
	if pdf is not None:
		p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")
	if cdf is not None:
		p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend="CDF")
	
	p.y_range.start = 0
	p.legend.location = "center_right"
	p.legend.background_fill_color = "#fefefe"
	p.xaxis.axis_label = 'Hours'
	p.yaxis.axis_label = 'Number of Charging Aggregators'
	p.grid.grid_line_color="white"
	return p
	
	
def write_output(filename,string):
	with open(SCENARIO_OUTPUT_FOLDER + filename,'w') as outfile:
		outfile.write(string)

'''
Bidding Profiles
'''

def basic_ev_fleet():

	bid_density = 100
	charge_start_mu = 2
	charge_start_sigma = 1
	charge_length_mu = 3
	charge_length_sigma = 6

def read_aggregator_profiles(profile_json_fname):
	
	'''Reads the aggregator input json file'''
	aggregator_profiles = {}
	
	with open(profile_json_fname,'r') as json_file:
	
		agg_data = json.load(json_file)
		for aggregator in agg_data['Aggregators']:
		
			aggregator_profiles[aggregator['Name']] = {}
			
			total_capacity = float(aggregator['Size']) * float(aggregator['Average_Vehicle_Capacity']) / 1000.0 #MWh
			charge_required = np.multiply(np.array(aggregator['Weekly_Usage']),total_capacity )
			
			aggregator_profiles[aggregator['Name']]['Charge_Required'] = charge_required
			
			aggregator_profiles[aggregator['Name']]['Charge_Required'] = float(aggregator['Avg_Daily_Demand']) * float(aggregator['Size']) / 1000.0 #MWh
			aggregator_profiles[aggregator['Name']]['bid_count'] = aggregator['bid_count']
			aggregator_profiles[aggregator['Name']]['charge_start_mu'] = aggregator['charge_start_mu']
			aggregator_profiles[aggregator['Name']]['charge_start_sigma'] = aggregator['charge_start_sigma']
			aggregator_profiles[aggregator['Name']]['charge_length_mu'] = aggregator['charge_length_mu']
			aggregator_profiles[aggregator['Name']]['charge_length_sigma'] = aggregator['charge_length_sigma']
			aggregator_profiles[aggregator['Name']]['Cost_Limit'] = float(aggregator['Cost_Limit'])



	return aggregator_profiles
	
'''
Bidding Model
'''
def write_bids(all_bids,no_dr_bids=None,output_filename=None):
	'''Writes the output csv files for the demand response bids from aggreagtors'''
	output_str = ('Bid_ID,Bidder_ID,zone,Day,start_time,end_time,' +
			'demand_min_total,demand_max_total,demand_min_timesteps,' + 
			'demand_max_timesteps,' + 'Cost_Limit' + '\n')
	
	for cur_bid in all_bids:
		output_str = output_str + str(cur_bid['Bid_ID']) + ','
		output_str = output_str + str(cur_bid['Bidder_ID']) + ','
		output_str = output_str + str(cur_bid['zone']) + ','
		output_str = output_str + str(cur_bid['Day']) + ','
		output_str = output_str + str(cur_bid['start_time']) + ','
		output_str = output_str + str(cur_bid['end_time']) + ','
		output_str = output_str + str(cur_bid['demand_min_total']) + ','
		output_str = output_str + str(cur_bid['demand_max_total']) + ','
		output_str = output_str + str(cur_bid['demand_min_timesteps']).replace(',',';') + ','
		output_str = output_str + str(cur_bid['demand_max_timesteps']).replace(',',';') + ','
		output_str = output_str + str(cur_bid['Cost_Limit'])
		output_str = output_str + '\n'
	
	'Will Write No Demand Response Case for Control'
	if(no_dr_bids is not None):
		output_str_no_dr = ('Bid_ID,Bidder_ID,zone,Day,start_time,end_time,' +
			'demand_min_total,demand_max_total,demand_min_timesteps,' + 
			'demand_max_timesteps,' + 'Cost_Limit'  + '\n')
	
		for cur_bid in no_dr_bids:
			output_str_no_dr = output_str_no_dr + str(cur_bid['Bid_ID']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['Bidder_ID']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['zone']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['Day']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['start_time']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['end_time']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['demand_min_total']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['demand_max_total']) + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['demand_min_timesteps']).replace(',',';') + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['demand_max_timesteps']).replace(',',';') + ','
			output_str_no_dr = output_str_no_dr + str(cur_bid['Cost_Limit'])
			output_str_no_dr = output_str_no_dr + '\n'
		
		write_output('No_DR_' + output_filename, output_str_no_dr)
		
	write_output(output_filename, output_str)
	
def build_day_bids(total_bid_count,starting_mu, starting_std, charging_mu, charging_std,cost_limit, min_charge_amt = None, day = None, bidder_id = 'Tony'):
	'''Build a histogram of a single day of bids'''
	global BID_ID_INDEX
	
	headers = ('Bid_ID,Bidder_ID,zone,Day,start_time,end_time,' +
			'demand_min_total,demand_max_total,demand_min_timesteps,' + 
			'demand_max_timesteps')
	
	if(day is None):
		day = 0
	if min_charge_amt is None:
		min_charge_amt = np.random.randint(1,1300)
	#min_charge_amt = np.random.randint(1,1300)
	all_t = []
	hour_ctr = {}
	all_bids = []
	
	no_dr_bids = []
	
	for i in range(24):
		hour_ctr[i] = 0
	
	lost = 0
	
	while len(all_bids) < total_bid_count:
	
		cur_bid = {}
		cur_bid['Bid_ID'] = BID_ID_INDEX
		cur_bid['Bidder_ID'] = bidder_id
		cur_bid['zone'] = np.random.randint(1,6)
		
		zone_random = random.random()
				
		if zone_random < 0.90:
			cur_bid['zone'] = np.random.randint(1,4)
		else:
			cur_bid['zone'] = 4
		
		
		
		starting_time = int(np.random.normal(starting_mu,starting_std))
		charging_length = np.random.normal(charging_mu,charging_std)
		end_time = (int(starting_time) + int(charging_length))
		
		charging_times = range(starting_time,end_time + 1)
		demand_min_timesteps = []
		demand_max_timesteps = []
		
		
		
		if starting_time >= 0 and end_time < 24 and starting_time < end_time :
			for cur_hr in charging_times:
				hour_ctr[cur_hr] = hour_ctr[cur_hr] + 1
				all_t.append(cur_hr)
				demand_min_timesteps.append(0)
				demand_max_timesteps.append(15000)
			
			cur_bid['start_time'] = starting_time
			cur_bid['end_time'] = end_time
			cur_bid['demand_min_timesteps'] = demand_min_timesteps
			cur_bid['demand_max_timesteps'] = demand_max_timesteps
			cur_bid['demand_max_total'] = 20000000
			cur_bid['demand_min_total'] = min_charge_amt
			cur_bid['Day'] = day + 1
			cur_bid['Cost_Limit'] = cost_limit
			
			#Ensures feasibility
			if np.sum(np.array(demand_max_timesteps)) > cur_bid['demand_min_total']:
				
				#add no DR and DR bids
				cur_no_dr = copy.deepcopy(cur_bid)
				cur_no_dr['end_time'] = cur_no_dr['start_time']
				cur_no_dr['demand_min_timesteps'] = [cur_no_dr['demand_min_total']]
				cur_no_dr['demand_max_timesteps'] = [cur_no_dr['demand_min_total']]
				
				all_bids.append(cur_bid)
				no_dr_bids.append(cur_no_dr)
				
				BID_ID_INDEX = BID_ID_INDEX + 1


	'''Plotting'''
	hour_hist, edges = np.histogram(np.array(all_t),list(hour_ctr.keys()))
	edges = np.add(edges, day * 24)
	mu, sigma = (starting_mu + charging_mu), np.sqrt(starting_std**2 + charging_std**2)
	x = np.array(list(range(0,24)))
	
	pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
	cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2
	
	
	p1 = make_plot( ("ID: %s Charging Hours Distribution (Normal) (μ=%s, σ=%s)" % (bidder_id, str(mu), str(sigma)) ) , 
					hour_hist, edges, list(range(0,24)), None, None)
	p2 = make_plot( ("ID: %s Charging Hours PDF and CDF (Normal) (μ=%s, σ=%s)" % (bidder_id, str(mu), str(sigma)) ) , 
					None, edges, list(range(0,24)), pdf, cdf)
	

	
	return hour_hist, edges, mu, sigma, [p1,p2], all_bids, no_dr_bids

def build_weekly_bids():
	
	output_file('histogram.html', title="histogram.py example")
	
	weekly_edges = []
	weekly_hourly_hist = []
	weekly_bids = []
	weekly_no_dr_bids = []
	for day in range(7):
		hour_hist, edges, mu, sigma, plots, day_bids, no_dr_bids = build_day_bids(400,2,1,3,6,day,'Fleet')
		weekly_edges = weekly_edges + list(np.array(edges)) 
		weekly_hourly_hist = weekly_hourly_hist + list(np.array(hour_hist))
		weekly_bids = weekly_bids + day_bids
		
		weekly_no_dr_bids = weekly_no_dr_bids + no_dr_bids
		
	
	week_plot = make_plot( ("Fleet Week Charging Hours Distribution " ) , 
					weekly_hourly_hist, weekly_edges, list(range(0,167)), None, None)
	
	residential_weekly_edges = []
	residential_weekly_hourly_hist = []
	residential_weekly_bids = []
	residential_weekly_no_dr_bids = []
	
	for day in range(7):
		if day == 0 or day == 6:
			hour_hist, edges, mu, sigma, plots, day_bids, no_dr_bids = build_day_bids(200,12,5,5,6,day,'Residential')
		else:
			hour_hist, edges, mu, sigma, plots, day_bids, no_dr_bids = build_day_bids(500,2,1,3,6,day,'Residential')
		residential_weekly_edges = residential_weekly_edges + list(np.array(edges)) 
		residential_weekly_hourly_hist = residential_weekly_hourly_hist + list(np.array(hour_hist))
		residential_weekly_bids = residential_weekly_bids + day_bids
		
		residential_weekly_no_dr_bids = residential_weekly_no_dr_bids + no_dr_bids
	
	
	residential_week_plot = make_plot( ("Residential Weekly Charging Hours Distribution" ) , 
					residential_weekly_hourly_hist, residential_weekly_edges, list(range(0,167)), None, None)
	
	
	write_bids(weekly_bids + residential_weekly_bids, weekly_no_dr_bids + residential_weekly_no_dr_bids)
	
	
	hour_hist2, edges2, mu2, sigma2, plots2, day_bids,no_dr_bids = build_day_bids(100,2,4,5,6, 4)
	
	res_hist1, res_edges1, res_mu1, res_sigma1, res_plots1, day_bids, no_dr_bids = build_day_bids(1000,1,1,2,2)
	res_hist2, res_edges2, res_mu2, res_sigma2, res_plots2, day_bids,no_dr_bids = build_day_bids(1000,18,1,2,4)
	
	
	res_total = make_plot( ("Charging Hours Distribution (Normal) (μ=%s, σ=%s)" % (str(mu), str(sigma)) ) , 
					np.add(res_hist1 , res_hist2), res_edges1, list(range(0,24)), None, None)
	
	show(gridplot([week_plot] + [residential_week_plot] + plots2 + res_plots1 + res_plots2 + [res_total], ncols=2, plot_width=600, plot_height=600))

	#print(hour_hist)
	#print(hour_hist2)
	#print(np.add(hour_hist, hour_hist2))

	
def build_aggregator_bids(run_name, aggregator_file_name=None, output_filename=None):
	''' 
	Build Bids from Input Data file
	Input File: aggregator_profiles.json
	
	'''
	if aggregator_file_name is None:
		aggregator_profiles = read_aggregator_profiles('aggregator_data/aggregator_profiles1.json')
	else:
		aggregator_profiles = read_aggregator_profiles(aggregator_file_name)
		
	output_file('aggregator_data\\' + run_name + '_Charging_Distribution.html', title="Aggregator Charging Frequency")
	
	
	all_bids = []
	all_no_dr_bids = []
	all_plots = []
	daily_plots = []
	
	for agg_name in aggregator_profiles.keys():
	
		cur_profile = aggregator_profiles[agg_name]
		edges = []
		hourly_hist = []
		
		if agg_name == 'Buses':
			'Bus Aggregator'
			for day in range(7):
				half_bid_count = int(cur_profile['bid_count'] / 2)
				
				cur_hour_hist1, cur_edge1, mu1, sigma1, day_plots1, day_bids1, no_dr_bids1 = build_day_bids(half_bid_count,cur_profile['charge_start_mu'][day],
																						  cur_profile['charge_start_sigma'][day],cur_profile['charge_length_mu'][day],
																						  cur_profile['charge_length_sigma'][day],cur_profile['Cost_Limit'],
																						  (cur_profile['Charge_Required'] / float(half_bid_count)), day,agg_name)
																						  
				cur_hour_hist2, cur_edge2, mu2, sigma2, day_plots2, day_bids2, no_dr_bids2 = build_day_bids(half_bid_count,np.add(cur_profile['charge_start_mu'][day],15),
																						  cur_profile['charge_start_sigma'][day],cur_profile['charge_length_mu'][day],
																						  cur_profile['charge_length_sigma'][day],cur_profile['Cost_Limit'],
																						   (cur_profile['Charge_Required'] / float(half_bid_count)), day,agg_name)
				
				cur_hour_hist = np.add(cur_hour_hist1, cur_hour_hist2)
				cur_edge = cur_edge1
				day_bids = day_bids1 + day_bids2
				no_dr_bids = no_dr_bids1 + no_dr_bids2
				
				edges = edges + list(np.array(cur_edge)) 
				hourly_hist = hourly_hist + list(np.array(cur_hour_hist))
				all_bids = all_bids + day_bids
				all_no_dr_bids = all_no_dr_bids + no_dr_bids
				
		else:
			'Standard Aggregator'
			for day in range(7):
				cur_hour_hist, cur_edge, mu, sigma, [day_plots_hist,day_plots_dist], day_bids, no_dr_bids = build_day_bids(cur_profile['bid_count'],cur_profile['charge_start_mu'][day],
																						  cur_profile['charge_start_sigma'][day],cur_profile['charge_length_mu'][day],
																						  cur_profile['charge_length_sigma'][day],cur_profile['Cost_Limit'],
																						  (cur_profile['Charge_Required'] / float(cur_profile['bid_count'])), day,agg_name)
				edges = edges + list(np.array(cur_edge)) 
				hourly_hist = hourly_hist + list(np.array(cur_hour_hist))
				all_bids = all_bids + day_bids
				all_no_dr_bids = all_no_dr_bids + no_dr_bids
				
				daily_plots.append(day_plots_hist)
				daily_plots.append(day_plots_dist)
	
		week_plot = make_plot( ("ID: " + agg_name + " Week Charging Hours Distribution " ) , 
								hourly_hist, edges, list(range(0,167)), None, None)
				
		all_plots.append(week_plot)
		
	show(gridplot(daily_plots, ncols=2, plot_width=800, plot_height=800))
	
	if output_filename is None:
		write_bids(all_bids, all_no_dr_bids,OUTPUT_FILE_NAME)
	else:
		write_bids(all_bids, all_no_dr_bids, output_filename)
	
	
def build_scenarios():
	
	global AGGREGATOR_DATA_FOLDER
	global SCENARIO_FOLDER
	global SCENARIO_OUTPUT_FOLDER
	
	SCENARIO_FOLDER = 'aggregator_data\\Scenarios_Updated\\'
	SCENARIO_OUTPUT_FOLDER = 'aggregator_data\\Generated_Data\\'
	
	#scenarios = ['low-day','low-night','med-day','med-night','high-day','high-night']
	scenarios = ['real-low-evening','real-low-morning','real-med-evening','real-med-morning','real-high-evening','real-high-morning']
	#scenarios = ['real-med-day']
	
	for run in scenarios:
		build_aggregator_bids(run, SCENARIO_FOLDER + run + '.json', run + '-bids.csv')
		
if __name__ == '__main__':
	#build_weekly_bids()
	#build_aggregator_bids()
	build_scenarios()
	
	