'''
Tony Shu
Combine Electric Market Outputs
'''

import numpy as np
import csv
import os.path

output_folder = 'simulation_outputs\\FinalResults\\'
date_strs = ['04-08-2020']
scenarios = ['default','real-high-evening', 'real-med-evening', 'real-low-evening', 'real-high-morning', 'real-med-morning', 'real-low-morning']
networks = ['L,L,11,621']

data_filenames = [ 'tab_dam_demand_response_load',
				   'tab_dam_generator_power',
				   'tab_dam_loads',
				   'tab_dam_prices',
				   'tab_dam_storage_unit_power',
				   'tab_dam_storage_unit_soc',
				   'tab_gen_outage',
				   'tab_rtm_generator_power',
				   'tab_rtm_prices'
				]

#combined_output_folder = 'simulation_outputs\\combined-outputs-default\\'
combined_output_folder = 'simulation_outputs\\FinalResults\\L,L,11,621-combined-results\\'

for cur_network in networks:
	for cur_date in date_strs:
		for cur_scenario in scenarios:
			
			#if ((cur_network == 'L,L,8,621,solar-added' or cur_network == 'L,L,8,621,wind-added') and cur_scenario != 'default') or cur_network == 'L,H,8,621':
			if True:
				run_folder = output_folder + '\\' + cur_scenario + '-' + cur_date + '-' + cur_network +'\\'
				dr_folder = run_folder + 'demand-response\\tableau_output\\'
				no_dr_folder = run_folder + 'no-demand-response\\tableau_output\\'
			
				for filename in data_filenames:
			
					dr_output_string = ''
				
					with open(dr_folder + filename + '.csv','r') as infile:
						#Get rid of CSV headers if already exists
						if os.path.isfile(combined_output_folder + filename + '_combined.csv'):
							dr_output_string = infile.readline()
						dr_output_string = infile.read()

				
					with open(combined_output_folder + filename + '_combined.csv','a+') as outfile:
						outfile.write(dr_output_string)
					
					no_dr_output_string = ''
				
					if cur_scenario != 'default':
						with open(no_dr_folder + filename + '.csv','r') as infile:
							#Get rid of CSV headers if already exists
							if os.path.isfile(combined_output_folder + filename + '_combined.csv'):
								no_dr_output_string = infile.readline()
							no_dr_output_string = infile.read()
					
						with open(combined_output_folder + filename + '_combined.csv','a+') as outfile:
							outfile.write(no_dr_output_string)
			