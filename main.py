import os
import numpy as np
import get_sgf
import class_tot
import matplotlib.pyplot as plt

# script contains code to read/present soundings and interpret layers/units
# code to plot in-phase correction diagrams was added at bottom
 
save_figs = False
plot_phase = True
plot_3phase = False

def get_file_list( base_dir ):
    return [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]

if __name__ == '__main__':
    image_folder = 'data/img'
# test files: site spesific
    if False:
        data_folder = 'data/Oysand'

# classified by characteristics (not units!) for comparison
        profile_A = ['_PR-A',[
            ['oysts39 20201022 1090.TOT',[0, 1.25, 3.73, 5.05, 12.80, 13.18, 18.00, 20.44, 21.850]],
            ['oysts24 20201022 1092.TOT',[0, 1.36, 3.82, 5.32, 12.60, 12.85, 17.52, 19.95, 21.825]],
            ['oysts29 20201021 1084.TOT',[0, 1.47, 3.83, 5.83, 12.23, 12.49, 17.15, 19.70, 21.825]],
            ['oysts35 20201023 1097.TOT',[0, 1.53, 3.83, 6.33, 11.82, 12.48, 16.88, 19.47, 21.825]],
            ['oysts31 20201023 1098.TOT',[0, 1.58, 3.80, 6.41, 11.27, 12.05, 16.19, 18.79, 21.825]]
            ]]
        NORMAL  = ['_NORM',[
            ['oysts45 20201021 1088.TOT',[0, 0.62, 1.40, 2.58, 3.65, 5.20, 6.55, 9.23, 12.36, 18.4,21.6, 21.8]],
            ['oysts41 20201021 1087.TOT',[0, 0.39, 1.38, 3.14, 3.82, 4.82, 6.77, 8.30, 12.37, 18.1,21.1,21.85]],
            ['oysts29 20201021 1084.TOT',[0, 0.62, 1.54, 2.91, 3.83, 5.40, 6.28, 7.81, 12.22, 17.4,21.1,21.825]],
            ['oysts47 20201021 1086.TOT',[0, 1.04, 1.66, 2.90, 4.69, 5.55, 6.40, 7.55, 10.68, 16.7,20.2,21.85]],
            ['oysts25 20201021 1085.TOT',[0, 1.21, 1.53, 2.61, 3.91, 5.57, 6.08, 7.96, 11.05, 16.4,19.8,21.9]] 
            ]]

        ALL  = ['_NORM',[
            ['oysts45 20201021 1088.TOT',[0, 0.62, 1.40, 2.58, 3.65, 5.20, 6.55, 9.23, 12.36, 18.40, 21.60, 21.8], "normal"],
            ['oysts41 20201021 1087.TOT',[0, 0.39, 1.38, 3.14, 3.82, 4.82, 6.77, 8.30, 12.37, 18.10, 21.10, 21.85], "normal"],
            ['oysts29 20201021 1084.TOT',[0, 0.62, 1.54, 2.91, 3.83, 5.20, 6.28, 7.81, 12.22, 17.40, 21.10, 21.825], "normal"],
            ['oysts47 20201021 1086.TOT',[0, 1.04, 1.66, 2.90, 4.69, 5.55, 6.40, 7.55, 10.68, 16.70, 20.20, 21.85], "normal"],
            ['oysts25 20201021 1085.TOT',[0, 1.21, 1.53, 2.61, 3.91, 5.57, 6.08, 7.96, 11.05, 16.40, 19.80, 21.9], "normal"],
            ['oysts24 20201022 1092.TOT',[0, 0.60, 1.5, 2.29, 3.63, 5.33, 6.65, 8.05, 12.55, 17.8, 20.6, 21.825], "slow"], #slow
            ['oysts31 20201023 1098.TOT',[0, 0.80, 1.7, 2.00, 3.60, 5.49, 6.0, 8.39, 11.30, 17.2, 19.70, 21.825], "slow"], #slow
            ['oysts27 20201023 1094.TOT',[0, 0.60, 1.55, 2.7, 3.63, 5.2, 6.3, 7.8, 12.1, 18.1, 20.9, 21.825], "very slow"], #very slow
            ['oysts46 20201023 1095.TOT',[0, 0.60, 1.7, 2.7, 3.6, 5.4, 6.5, 7.4, 12.1, 17.7, 21.4, 22.0], "very slow"], #very slow
            ['oysts35 20201023 1097.TOT',[0, 0.80, 1.7, 2.00, 3.60, 5.8, 8, 9.3, 11.80, 17.1, 19.9, 21.9], "fast"], # fast
            ['oysts39 20201022 1090.TOT',[0, 0.62, 1.54, 2.91, 3.83, 4.80, 7.5, 9.0, 12.8, 19.2, 21.3, 21.850], "fast"], # fast
            ['oysts26 20201023 1096.TOT',[0, 0.62, 1.54, 2.91, 3.83, 5.3, 6.2, 7.6, 12.22, 17.40, 21.05, 21.825], "very fast"], # very fast
            ['oysts36 20201023 1093.TOT',[0, 0.62, 1.54, 2.91, 3.83, 5.20, 6.28, 7.6, 11.8, 17.20, 20.8, 21.825], "very fast"] # very fast
            ]]

# classified by units       
        speed_norm  = ['_NORM',[
            ['oysts45 20201021 1088.TOT',[0,1.4,5.2,9.8,12.4,18.4,21.6, 21.8]], #[0,1.4,5.2,9.8,12.4,18.2,21.8]
            ['oysts41 20201021 1087.TOT',[0,1.4,4.7,9.4,12.4,18.1,21.1,21.85]],
            ['oysts29 20201021 1084.TOT',[0,1.4,5.35,8.1,12.2,17.4,21.1,21.825]],
            ['oysts47 20201021 1086.TOT',[0,1.6,5.60,6.2,10.7,16.7,20.2,21.85]],  # [0,1.6,5.55,8.1,10.7,16.7,20.5,21.85]
            ['oysts25 20201021 1085.TOT',[0,1.5,5.6,8.15,11,16.4,19.8,21.9]] 
            ]]
        speed_slow  = ['_SLOW_1_kriging',[
            ['oysts24 20201022 1092.TOT',[0, 1.5, 5.35, 9.0, 12.5, 17.5, 21.4, 21.825]],
            ['oysts31 20201023 1098.TOT',[0, 1.57, 6.4, 7.25, 11.2, 16.4, 20.1, 21.825]]
            ]]
        speed_sloow = ['_SLOW_2_kriging',[
            ['oysts27 20201023 1094.TOT',[0, 1.45, 5.34, 8.1, 12.1, 17.5, 20.75, 21.825]],
            ['oysts46 20201023 1095.TOT',[0, 1.65, 5.6, 7.5, 11.65, 17.6, 21.2, 21.850]]
            ]]
        speed_fast  = ['_FAST_1_kriging',[
            ['oysts35 20201023 1097.TOT',[0, 1.52, 6.0, 7.4, 11.6, 16.9, 20.7, 21.825]],
            ['oysts39 20201022 1090.TOT',[0, 1.25, 5.0, 10.4, 12.75, 18.5, 21.5, 21.850]]
            ]]
        speed_faast = ['_FAST_2_kriging',[
            ['oysts26 20201023 1096.TOT',[0, 1.5, 5.6, 7.9, 12.1, 17.5, 21.1, 21.775]],
            ['oysts36 20201023 1093.TOT',[0, 1.4, 5.3, 7.9, 11.85, 17.3, 20.5, 21.825]]
            ]]

    else:
        data_folder = 'data/Kjellstad'
# classified by units  
        speed_norm  = ['_NORM',[
            ['KJTS01-Tot.std',[ 0.1, 0.84, 2.31, 8.02, 8.76, 15.16, 17.26, 20.02], "normal"],
            ['KJTS02-Tot.std',[ 0.1, 1.03, 2.15, 7.97, 8.55, 14.85, 17.05, 20.02], "normal"],
            ['KJTS03-Tot.std',[ 0.1, 1.26, 2.52, 8.07, 8.80, 15.25, 17.33, 20.02], "normal"],
            ['KJTS04-Tot.std',[ 0.1, 0.82, 2.55, 7.60, 8.26, 15.07, 17.16, 20.02], "normal"],
            ['KJTS05-Tot.std',[ 0.1, 1.04, 2.49, 8.54, 9.31, 15.40, 17.61, 20.21], "normal"]
            ]]
        speed_slow  = ['_SLOW_1',[
            ['KJTS11-Tot.std',[]],
            ['KJTS13-Tot.std',[]]
            ]]
        speed_sloow = ['_SLOW_2',[
            ['KJTS06-Tot.std',[]],
            ['KJTS08-Tot.std',[]]
            ]]
        speed_fast  = ['_FAST_1',[
            ['KJTS10-Tot.std',[]],
            ['KJTS12-Tot.std',[]]
            ]]
        speed_faast = ['_FAST_2',[
            ['KJTS07-Tot.std',[]],
            ['KJTS09-Tot.std',[]]
            ]]

        ALL  = ['_ALL',[
            ['KJTS01-Tot.std',[ 0.1, 0.84, 2.31, 8.02, 8.76, 15.16, 17.26, 20.02], "normal"],
            ['KJTS02-Tot.std',[ 0.1, 1.03, 2.15, 7.97, 8.55, 14.85, 17.05, 20.02], "normal"],
            ['KJTS03-Tot.std',[ 0.1, 1.26, 2.52, 8.07, 8.80, 15.25, 17.33, 20.02], "normal"],
            ['KJTS04-Tot.std',[ 0.1, 0.82, 2.55, 7.60, 8.26, 15.07, 17.16, 20.02], "normal"],
            ['KJTS05-Tot.std',[ 0.1, 1.04, 2.49, 8.54, 9.31, 15.40, 17.61, 20.21], "normal"],
            ['KJTS11-Tot.std',[ 0.1, 0.72, 2.32, 7.96, 8.67, 15.09, 17.19, 20.02], "slow"],
            ['KJTS13-Tot.std',[ 0.1, 0.90, 2.49, 8.34, 9.07, 15.39, 17.46, 20.01], "slow"],
            ['KJTS06-Tot.std',[ 0.1, 1.68, 2.53, 8.01, 8.71, 15.22, 17.33, 20.03], "very slow"],
            ['KJTS08-Tot.std',[ 0.1, 0.84, 2.51, 8.18, 8.92, 15.20, 17.29, 20.02], "very slow"],
            ['KJTS10-Tot.std',[ 0.1, 0.87, 2.51, 8.15, 8.79, 15.05, 17.14, 20.03], "fast"],
            ['KJTS12-Tot.std',[ 0.1, 0.88, 2.66, 8.18, 8.93, 15.26, 17.37, 20.74], "fast"],
            ['KJTS07-Tot.std',[ 0.1, 1.09, 2.50, 8.06, 8.79, 15.22, 17.33, 20.05], "very fast"],
            ['KJTS09-Tot.std',[ 0.1, 0.75, 2.60, 8.24, 8.98, 15.33, 17.37, 20.05], "very fast"]
            ]]
    tot_files = np.array( get_file_list( data_folder ) )

# select set for this run
    soundings = ALL #speed_norm #profile_A# speed_norm, speed_fast, speed_faast, speed_sloow, speed_slow
    show_figs = save_figs
    total_soundings = []

    for sounding_data in soundings[1]:
        temp_tot_data = get_sgf.read_tot( os.path.join( data_folder, sounding_data[0] ), sounding_nr=0 )
        
        if len(sounding_data) < 3:
            sounding_data.append("")
        
        total_soundings.append( class_tot.tot( sgf_data=temp_tot_data, layers=sounding_data[1], comment=sounding_data[2] ) )
        #total_soundings[-1].get_stats()
        if save_figs:
            img_name = 'TOT_' + total_soundings[-1].get_name() + soundings[0] + '.png'
            full_path = os.path.join(image_folder, img_name)
            total_soundings[-1].to_figure( 
                color_figure=True, 
                filename=full_path,
                plot_layers = True,
                show_layer_analysis=True
                )

        elif show_figs:
            total_soundings[-1].to_figure( color_figure=True )

if plot_phase:
    names = []
    D_unmasked =[]
    A_unmasked = []
    D_masked =[]
    A_masked = []

    for s in total_soundings:
        names.append(s.get_name() + " (" + s.get_comment() + ")")
        D, A = s.get_D_F_DT()
        D_unmasked.append(D)
        A_unmasked.append(A)
        
        s.apply_mask( [ 0.1, 0.84, 2.31, 8.02, 8.76, 15.16, 17.26, 20.02] ) #[0,1.4,5.35,8.1,12.2,17.4,21.1]

        D, A = s.get_D_F_DT()
        D_masked.append(D)
        A_masked.append(A)
        if False:
            s.set_layers( layers=[0,1.4,5.35,8.1,12.2,17.4,21.1,21.825] ) #from oysts29
            s.to_figure( color_figure=True, plot_layers = True )

    if plot_3phase: # plot three charts
        plt.figure(figsize=(7, 4))
        ax0 = plt.subplot(131)
        ax1 = plt.subplot(132)
        ax2 = plt.subplot(133)
        for n,D,A,DM,AM in zip( names, D_unmasked, A_unmasked, D_masked, A_masked ):
            delta_D = [D1 - D2 for (D1, D2) in zip(DM, D)]
            ax0.plot(A, D, label=n)
            ax1.plot(AM, DM, label=n)
            ax2.plot(delta_D, D, label=n)

        ax0.set_ylim( 0, 20 )
        ax1.set_ylim( 0, 20 )
        ax2.set_ylim( 0, 20 )
        ax2.set_xlim( -1.5, 1.5 )

        ax0.set_title('no correction')
        ax0.set_ylabel('Depth (m)', multialignment='center')
        ax0.set_xlabel('Push force (kN)')
        ax1.set_title('in-phase correction')
        ax1.set_xlabel('Push force (kN)')
        ax2.set_title('correction factor')
        ax2.set_xlabel('Depth shift (m)')

        ax0.invert_yaxis()
        ax0.grid( True )
        ax1.invert_yaxis()
        ax1.grid( True )
        ax2.invert_yaxis()
        ax2.grid( True )
        leg = plt.legend(loc='best', fancybox=True)
        plt.show()
    else:
        plt.figure(figsize=(7, 4))
        ax0 = plt.subplot(111)
        for n,D,A,DM,AM in zip( names, D_unmasked, A_unmasked, D_masked, A_masked ):
            delta_D = [D1 - D2 for (D1, D2) in zip(DM, D)]
            ax0.plot(AM, DM, label=n)
        ax0.set_ylim( 0, 20 )
        ax0.set_xlim( 0, 30 )
        ax0.set_ylabel('Depth (m)', multialignment='center')
        ax0.set_xlabel('phase-corrected push force (kN)')
        ax0.invert_yaxis()
        ax0.grid( True )
        leg = plt.legend(loc='best', fancybox=True)
        plt.show()