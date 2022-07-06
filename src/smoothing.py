import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.oned_filter import OneDimensionalFilter



# create main function
def main():

  frames = []
  times = []
  distances = []
  filtered = []

  # read csv file
  with open("../track_1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0


    

    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        elif line_count == 1:
            dist_filter = OneDimensionalFilter(float(row[2]))
            line_count += 1
        else:
            frame = float(row[0])
            dt = float(row[1])
            d = float(row[2])

            dist_filter.update(d, dt)
            
            frames.append(frame)
            times.append(dt)
            distances.append(d)
            filtered.append(dist_filter.x)
            line_count += 1


    print(f'Processed {line_count} frames.')


  fig, ax = plt.subplots()
  line1, = ax.plot(frames, distances, label = "Collision distance")
  line2, = ax.plot(frames, filtered, label='Filtered')
  ax.legend()
  plt.show()

  # print(frames)
  # print(distances)
  # plot the data
  # plt.plot(frames, distances, label = "Collision distance")
  # plt.xlabel('frame')
  # plt.ylabel('dist (m)')
  # plt.show()
  # cv2.waitKey(0)




if __name__ == "__main__":
    main()