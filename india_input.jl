# Tony Shu
# Reading India Data Files
# Mostly Generators are returned

using CSV
using DataFrames

function read_india_fuels(data_directory, filename="Fuels_data.csv")
	# Header Names
	#'fuel_indices, Fuel, Cost_per_MMBtu, CO2_content_tons_perMMBtu'
	
	df = CSV.read(data_directory * filename, copycols=true)
	cp_df = deepcopy(df)
	
	fuels = Dict()
	
	# Builds the fuels Dictionary
	for i in 1:size(df, 1)
		fuels[df[i,:].Fuel] = cp_df[i, :]
	end
	
	return fuels
end


function read_india_generators(data_directory, generator_filename="Generators_data.csv" )
	# Header Names
	#'R_ID,zone,voltage_level,Resource,RENEW,THERM,DISP,NDISP,STOR,DR,HEAT,NACC,HYDRO,VRE,'
	#'Commit,Min_Share,Max_Share,Existing_Cap_MW,New_Build,Cap_size,Max_Cap_MW,Min_Cap_MW,'
	#'Min_Share_percent,Max_Share_percent,Inv_cost_per_MWyr,Inv_cost_per_Mwhyr,Fixed_OM_cost_per_MWyr,'
	#'Var_OM_cost_per_MWh,Externality_cost_MWh,Start_cost,Start_fuel_MMBTU_per_start,Heat_rate_MMBTU_per_MWh,'
	#'Fuel,Min_power,Self_disch,Eff_up,Eff_down,Ratio_power_to_energy,Max_DSM_delay,Ramp_Up_percentage,'
	#'Ramp_Dn_percentage,Up_time,Down_time,NACC_Eff,NACC_Peak_to_Base,Reg_Up,Reg_Dn,Rsv_Up,Rsv_Dn,Reg_Cost,'
	#'Rsv_Cost,Fixed_OM_cost_per_MWhyr,Var_OM_cost_per_MWh_in,Hydro_level'
	
	df = CSV.read(data_directory * generator_filename, copycols=true)
	cp_df = deepcopy(df)
	
	# INITIALIZE NEW HEADERS IN DATAFRAME
	cp_df[!, :Marginal_cost] = Array{Float64}(undef, size(df,1))
	cp_df[!, :Num_generators] = Array{Int64}(undef, size(df,1))
	
	generators = Dict()
	fuels = read_india_fuels(data_directory)
	
	# Builds the generators Dictionary
	for i in 1:size(df, 1)
		generators[df[i,:].R_ID] = cp_df[i,:]
		
		#Existing Capacity conversion [ GW -> MW ]
		generators[df[i,:].R_ID].Existing_Cap_MW = float(generators[df[i,:].R_ID].Existing_Cap_MW) * 1000.0
		
		#Var OM cost conversion [ 1 / kWh -> 1 / MWh ]
		generators[df[i,:].R_ID].Var_OM_cost_per_MWh = float(generators[df[i,:].R_ID].Var_OM_cost_per_MWh) * 1000.0
		
		#Heat Rate MMBTU conversion [ 1 / kWh -> 1 / MWh ]
		generators[df[i,:].R_ID].Heat_rate_MMBTU_per_MWh = float(generators[df[i,:].R_ID].Heat_rate_MMBTU_per_MWh) * 1000.0
		
		#Start cost conversion [ 1 / kW -> 1 / MW ]
		generators[df[i,:].R_ID].Start_cost = float(generators[df[i,:].R_ID].Start_cost) * 1000.0
		
		
		#Float conversion to prevent type errors
		generators[df[i,:].R_ID].Eff_up = float(generators[df[i,:].R_ID].Eff_up)
		generators[df[i,:].R_ID].Eff_down = float(generators[df[i,:].R_ID].Eff_down)
		generators[df[i,:].R_ID].Hydro_level = float(generators[df[i,:].R_ID].Hydro_level)
		generators[df[i,:].R_ID].Ramp_Up_percentage = float(generators[df[i,:].R_ID].Ramp_Up_percentage)
		generators[df[i,:].R_ID].Ramp_Dn_percentage = float(generators[df[i,:].R_ID].Ramp_Dn_percentage)
		generators[df[i,:].R_ID].Self_disch =  float(generators[df[i,:].R_ID].Self_disch)
		
		#Int conversion to prevent type errors
		generators[df[i,:].R_ID].Up_time = Int(floor(generators[df[i,:].R_ID].Up_time))
		generators[df[i,:].R_ID].Down_time = Int(floor(generators[df[i,:].R_ID].Down_time))
		
		#Calculate the Start up cost
		generators[df[i,:].R_ID].Start_cost = generators[df[i,:].R_ID].Start_cost * generators[df[i,:].R_ID].Existing_Cap_MW
		
		#Calculate the Marginal Cost per MWh - NEED TO READ IN FUELS DATA
		generators[df[i,:].R_ID].Marginal_cost = (generators[df[i,:].R_ID].Var_OM_cost_per_MWh + generators[df[i,:].R_ID].Heat_rate_MMBTU_per_MWh * fuels[generators[df[i,:].R_ID].Fuel].Cost_per_MMBtu)
		
		#Calculate the cluster size (Number of generators) with Cap size conversion [ GW -> MW ]
		generators[df[i,:].R_ID].Num_generators = Int(floor(generators[df[i,:].R_ID].Existing_Cap_MW / (generators[df[i,:].R_ID].Cap_size * 1000.0 ) ))
		
	end
	
	return generators
end


function read_india_network(data_directory, filename="Network.csv")
	# Header Names
	#'Names,Share of zonal demand,Network_zones,VRE_Share,DistrZones,CO_2_Max_ton_MWh,'
	#'InZoneLossFact_Int,InZoneLossFact_W,InZoneLossFact_I,InZoneLossFact_N,VRE_Share,'
	#'Share_in_MV,DistrLossFact_LV_Net_Quad,DistrLossFact_MV_Net_Linear,DistrLossFact_LV_Total_Linear,'
	#'Predicted Average Loss at Peak,Assumed Distr. Headroom,Distr_Max_Inject,Distr_Max_Withdraw,'
	#'Distr_Inject_Max_Reinforcement_MW,Distr_Withdraw_Max_Reinforcement_MW,Distr_MV_Reinforcement_Cost_per_MW_yr,'
	#'Distr_LV_Reinforcement_Cost_per_MW_yr,DistrMarginFact_LV_Linear,DistrMarginFact_LV_Quad,'
	#'DistrMarginFact_MV_Linear,DistrMargin_MV_Max,DistrMargin_MV_DiscountFact,Network_lines,'
	#'Link_names,z1,z2,z3,z4,z5,Line_Loss_Percentage,Line_Max_Flow_MW,Initial_by_2015,Line_Max_Reinforcement_MW,'
	#'Line_Reinforcement_Cost_per_MW_yr,Line_Voltage_kV,Line_Resistance_ohms,Line_X_ohms,'
	#'Line_R_ohms,Thetha_max,Peak_Withdrawal_Hours,Peak_Injection_Hours'
	
	df = CSV.read(data_directory * filename, copycols=true)
	cp_df = deepcopy(df)
	
	links = Dict()
	zones = ["1","2","3","4","5"]
	
	for i in 1:size(df, 1)
	
		# Locate bus0 and bus1
		bus0 = ""
		bus1 = ""
		
		# Locate bus0 by value < 0
		if df[i,:].z1 < 0
			bus0 = "1"
		elseif df[i,:].z2 < 0
			bus0 = "2"
		elseif df[i,:].z3 < 0
			bus0 = "3"
		elseif df[i,:].z4 < 0
			bus0 = "4"
		elseif df[i,:].z5 < 0
			bus0 = "5"
		end
		
		# Locate bus1 by value > 0
		if df[i,:].z1 > 0
			bus1 = "1"
		elseif df[i,:].z2 > 0
			bus1 = "2"
		elseif df[i,:].z3 > 0
			bus1 = "3"
		elseif df[i,:].z4 > 0
			bus1 = "4"
		elseif df[i,:].z5 > 0
			bus1 = "5"
		end
		
		links[(bus0,bus1)] = cp_df[i,:]
		
		#Line Capacity conversion [ GW -> MW ]
		links[(bus0,bus1)].Line_Max_Flow_MW = float(links[(bus0,bus1)].Line_Max_Flow_MW) * 1000.0
	end

	return links, zones
end

function read_india_loads(data_directory, filename="Load_data.csv")
	# Header Names
	#'Voll,Demand_segment,Cost_of_demand_curtailment_perMW,'
	#'Max_demand_curtailment,Subperiods,Hours_per_period,Sub_Weights,'
	#'Time_index,Load_MW_z1,Load_MW_z2,Load_MW_z3,Load_MW_z4,Load_MW_z5'
	
	df = CSV.read(data_directory * filename, copycols=true)
	cp_df = deepcopy(df)
	
	loads = Dict()
	zone_loads = Dict("1"=>Array{Float64}(undef,0), 
				  "2"=>Array{Float64}(undef,0), 
				  "3"=>Array{Float64}(undef,0), 
				  "4"=>Array{Float64}(undef,0), 
				  "5"=>Array{Float64}(undef,0) )
	
	# Create the zonal load dictionary and load dictionary
	for i in 1:size(df, 1)
		
		loads[df[i,:].Time_index] = cp_df[i,:]
		
		# Load conversion [ GW -> MW ]
		loads[df[i,:].Time_index].Load_MW_z1 = float(loads[df[i,:].Time_index].Load_MW_z1) * 1000.0
		loads[df[i,:].Time_index].Load_MW_z2 = float(loads[df[i,:].Time_index].Load_MW_z2) * 1000.0
		loads[df[i,:].Time_index].Load_MW_z3 = float(loads[df[i,:].Time_index].Load_MW_z3) * 1000.0
		loads[df[i,:].Time_index].Load_MW_z4 = float(loads[df[i,:].Time_index].Load_MW_z4) * 1000.0
		loads[df[i,:].Time_index].Load_MW_z5 = float(loads[df[i,:].Time_index].Load_MW_z5) * 1000.0
		
		append!(zone_loads["1"],loads[df[i,:].Time_index].Load_MW_z1)
		append!(zone_loads["2"],loads[df[i,:].Time_index].Load_MW_z2)
		append!(zone_loads["3"],loads[df[i,:].Time_index].Load_MW_z3)
		append!(zone_loads["4"],loads[df[i,:].Time_index].Load_MW_z4)
		append!(zone_loads["5"],loads[df[i,:].Time_index].Load_MW_z5)
		
	end
	
	return loads, zone_loads
end

function read_india_variability(data_directory, filename="Generators_variability.csv")
	# Header Names
	#'Time_index,Solar/1,Wind/1,Biomass/1,Mini Hydro/1,'
	#'Pumped Hydro Storage/1,Hydro Reservoir/1,Hydro Run of River/1,...'
	
	df = CSV.read(data_directory * filename, copycols=true)
	cp_df = deepcopy(df)
	
	variability = Dict()
	
	for i in 1:size(df, 1)
		variability[df[i,:].Time_index] = cp_df[i,:]
	end

	return variability
end

read_india_generators("india_data_cases\\tony-2\\L,H,8,1275\\")
read_india_network("india_data_cases\\tony-2\\L,H,8,1275\\")
read_india_loads("india_data_cases\\tony-2\\L,H,8,1275\\")
read_india_variability("india_data_cases\\tony-2\\L,H,8,1275\\")








