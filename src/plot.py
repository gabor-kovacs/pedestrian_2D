import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.oned_filter import OneDimensionalFilter



# create main function
def main():

  frame_list = []
  d_list = []
  s_list = []
  s_gaze_list = []
  s_dist_list = []
  s_loc_list = []
  # read csv file
  with open("../track_1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) # skip header

    for row in csv_reader:
      frame = float(row[0])
      d = float(row[1])
      s = float(row[2])
      s_gaze = float(row[3])
      s_dist = float(row[4])
      s_loc = float(row[5])

      frame_list.append(frame)
      d_list.append(d)
      s_list.append(s)
      s_gaze_list.append(s_gaze)
      s_dist_list.append(s_dist)
      s_loc_list.append(s_loc)



    fig, axs = plt.subplots(2)

    axs[0].set_xlim([10, 170])
    axs[1].set_xlim([10, 170])

    # axs[0].plot(track.frames, track.collision_distances, label = "Collision distance (d)")
    # axs[1].plot(track.frames, track.safety_levels, label = "Safety level (s)")
    # axs[2].plot(track.frames, track.awareness_levels, label = "Awareness level (s_gaze)")
    # axs[3].plot(track.frames, track.collision_levels, label = "Distance safety level (s_dist)")
    # axs[4].plot(track.frames, track.location_levels, label = "Location safety level (s_loc)")

    axs[0].plot(frame_list, d_list, label = "Collision distance (d)", color="black")
    axs[1].plot(frame_list, s_list, label = "Safety level (s)")
    axs[1].plot(frame_list, s_gaze_list, label = "Awareness level ($s_{gaze}$)")
    axs[1].plot(frame_list, s_dist_list, label = "Distance safety level ($s_{dist}$)")
    axs[1].plot(frame_list, s_loc_list, label = "Location safety level ($s_{loc}$)")

    # plt.axvline(x=150)
    # plt.axvline(x=170)
    # plt.axvline(x=185)
    # plt.axvline(x=287)
    axs[0].set_xlabel("frame")
    axs[1].set_xlabel('frame')
    axs[0].set_ylabel("distance [m]")
    axs[1].set_ylabel("level [-]")

    # # naming the x axis
    # plt.xlabel('frame')
    # # naming the y axis
    # plt.ylabel('dist [m] / level [-]')
    # # giving a title to my graph
    # plt.title(f'Track {track.track_id}')
    # # show a legend on the plot
    fig.subplots_adjust(bottom=0.25, wspace=0.33)
    plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2))

    # plt.plot( track.x_values, track.z_values, label = "Trajectory")
    plt.show()


if __name__ == "__main__":
    main()