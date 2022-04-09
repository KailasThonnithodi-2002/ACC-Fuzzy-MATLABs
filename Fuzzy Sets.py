# Import libraries required
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Determine ranges for the graphs
x_SpeedRel = np.arange(-15, 20, 5)
x_DistanceFrom = np.arange(-30, 5, 5)
x_input = np.arange(-15, 20, 5)

# Generate fuzzy membership functions 
# Relative Speed Declaration
SpeedRel_Slower = fuzz.trimf(x_SpeedRel, [-15, -15, 0])
SpeedRel_Same = fuzz.trimf(x_SpeedRel, [-5, 0, 5])
SpeedRel_Faster = fuzz.trimf(x_SpeedRel, [0, 15, 15])

# Relative Distance Declaration
DistanceFrom_Far = fuzz.trimf(x_DistanceFrom, [-30, -30 , -15])
DistanceFrom_Safe = fuzz.trimf(x_DistanceFrom, [-20, -15, -10])
DistanceFrom_Close = fuzz.trimf(x_DistanceFrom, [-15, 0, 0])

# Input Declaration
input_BrakeHard = fuzz.trimf(x_input, [-15, -15, -7.5])
input_BrakeSoftly = fuzz.trimf(x_input, [-7.5, -3.75, -0])
input_AccelSlowly = fuzz.trimf(x_input, [0, 3.75, 7.5])
input_AccelModFast = fuzz.trimf(x_input, [7.5, 15, 15])
input_Nothing = fuzz.trimf(x_input, [-5, 0, 5])


# Visualize these universes and membership functions - Refer to matplot documentation for more info (https://matplotlib.org/stable/index.html)
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))
ax0.plot(x_SpeedRel, SpeedRel_Slower, 'b', linewidth=1.5, label='Slower')
ax0.plot(x_SpeedRel, SpeedRel_Same, 'g', linewidth=1.5, label='Same Speed')
ax0.plot(x_SpeedRel, SpeedRel_Faster, 'r', linewidth=1.5, label='Faster')
ax0.set_title('Speed relative to car in front.')
ax0.legend()
ax1.plot(x_DistanceFrom, DistanceFrom_Far, 'b', linewidth=1.5, label='Far')
ax1.plot(x_DistanceFrom, DistanceFrom_Safe, 'g', linewidth=1.5, label='Safe Distance')
ax1.plot(x_DistanceFrom, DistanceFrom_Close, 'r', linewidth=1.5, label='Close/Extremely Close')
ax1.set_title('Distance from Car in Front')
ax1.legend()
ax2.plot(x_input, input_BrakeHard, 'b', linewidth=1.5, label='Brake Hard')
ax2.plot(x_input, input_BrakeSoftly, 'g', linewidth=1.5, label='Brake Softly')
ax2.plot(x_input, input_AccelSlowly, 'purple', linewidth=1.5, label='Accelerate slowly')
ax2.plot(x_input, input_AccelModFast, 'r', linewidth=1.5, label='Accelerate Moderately fast')
ax2.plot(x_input, input_Nothing, 'orange', linewidth=1.5, label='Maintain Speed')
ax2.set_title('Brake or Accelerate?')
ax2.legend()
# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tight_layout()



#qual = speed relativity
#serv = distance from car


# We need the activation of our fuzzy membership functions at these values

speed_level_lo = fuzz.interp_membership(x_SpeedRel, SpeedRel_Slower, -10)
speed_level_md = fuzz.interp_membership(x_SpeedRel, SpeedRel_Same, -10)
speed_level_hi = fuzz.interp_membership(x_SpeedRel, SpeedRel_Faster, -10)

distance_level_lo = fuzz.interp_membership(x_DistanceFrom, DistanceFrom_Far, -25)
distance_level_md = fuzz.interp_membership(x_DistanceFrom, DistanceFrom_Safe, -25)
distance_level_hi = fuzz.interp_membership(x_DistanceFrom, DistanceFrom_Close, -25)


active_rule1 = np.fmax(speed_level_lo, distance_level_lo)
active_rule2 = np.fmax(speed_level_md, distance_level_lo)
active_rule3 = np.fmax(speed_level_hi, distance_level_lo)


active_rule4 = np.fmax(speed_level_lo, distance_level_md)
active_rule5 = np.fmax(speed_level_md, distance_level_md)
active_rule6 = np.fmax(speed_level_hi, distance_level_md)

active_rule7 = np.fmax(speed_level_lo, distance_level_hi)
active_rule8 = np.fmax(speed_level_md, distance_level_hi)
active_rule9 = np.fmax(speed_level_hi, distance_level_hi)







# Now we take our rules and apply them.

# Now we apply this by clipping the top off the corresponding output
# membership function with `np.fmin`
input_activation_lo = np.fmin(active_rule1, input_BrakeHard) 


input_activation_md = np.fmin(active_rule8 or active_rule6 or active_rule7, input_BrakeSoftly)


input_activation_hi = np.fmin(active_rule1, input_AccelModFast)



input_activation_medium = np.fmin(active_rule2 or active_rule4, input_AccelSlowly)


input_activation_nothing = np.fmin(active_rule3 or active_rule5, input_Nothing)

input0 = np.zeros_like(x_input)



# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_input, input0, input_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_input, input_BrakeHard, 'b', linewidth=0.5, linestyle='--', )

ax0.fill_between(x_input, input0, input_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_input, input_BrakeSoftly, 'g', linewidth=0.5, linestyle='--')

ax0.fill_between(x_input, input0, input_activation_hi, facecolor='purple', alpha=0.7)
ax0.plot(x_input, input_AccelSlowly, 'purple', linewidth=0.5, linestyle='--')

ax0.fill_between(x_input, input0, input_activation_medium, facecolor='r', alpha=0.7)
ax0.plot(x_input, input_AccelModFast, 'r', linewidth = 0.5, linestyle = '--')

ax0.fill_between(x_input, input0, input_activation_nothing, facecolor='orange', alpha=0.7)
ax0.plot(x_input, input_AccelModFast, 'orange', linewidth = 0.5, linestyle = '--')

ax0.set_title('Output membership activity')


# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tight_layout()
    
# Aggregate all three output membership functions together
aggregated = np.fmax(input_activation_md, np.fmax(input_activation_lo, input_activation_hi))
# Calculate defuzzified result
input = fuzz.defuzz(x_input, aggregated, 'centroid')
input_activation = fuzz.interp_membership(x_input, aggregated, input) # for plot
# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(x_input, input_BrakeHard, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_input, input_BrakeSoftly, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_input, input_AccelSlowly, 'purple', linewidth=0.5, linestyle='--')
ax0.plot(x_input, input_AccelModFast, 'r', linewidth = 0.5, linestyle = '--')
ax0.plot(x_input, input_Nothing, 'orange', linewidth=0.5, linestyle = '--')
ax0.fill_between(x_input, input0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([input, input], [0, input_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')
# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tight_layout()
    
