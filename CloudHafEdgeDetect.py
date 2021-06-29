import cv2
import numpy as np
import skvideo.io
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import sqlite3

detect_on = True
hough_on = True
grid_on = True
horizon_on = False
circle_detection_on = True
line_detection_on = True
detect_only_clouds = True

parts_count = 4
detection_edge = 3
frame_size = 1024
frame_counter = 0
frame_skip = 1
horizon_offset = 0

write_counter = 100
# inicijuojama duomenų bazė
conn = sqlite3.connect('clouds.db')
c = conn.cursor()
# Sukuriame reikiamą lentelę saugoti duomenys
c.execute('''CREATE TABLE IF NOT EXISTS cumulus_expert
                 (id INTEGER PRIMARY KEY, filename text, image_cropped_name text, x_coord text, y_coord text, lines_count  text, circles_count text, is_cumulus text)''')
conn.close()

def cloud_detect(r, g, b):
    if detect_only_clouds:
        return  g>120 and b>120 and r>120 and r/b>0.6
    else:
        return True

def horizon_detect(r, g, b):
    return (b < 100 and (g/b > 0.8 or r/b > 0.8))


circle_filter = []
for i in range(parts_count*parts_count):

    circle_filter.append(KalmanFilter(dim_x=2, dim_z=1))
    circle_filter[i] = KalmanFilter(dim_x=2, dim_z=1)

    circle_filter[i].x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    circle_filter[i].F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    circle_filter[i].H = np.array([[1.,0.]])    # Measurement function
    circle_filter[i].P *= 1000.                 # covariance matrix
    circle_filter[i].R = 5                      # state uncertainty
    dt = 0.1
    circle_filter[i].Q = Q_discrete_white_noise(2, dt, .1) # process uncertainty

line_filter = []
for i in range(parts_count*parts_count):
    
    line_filter.append(KalmanFilter(dim_x=2, dim_z=1))
    line_filter[i] = KalmanFilter(dim_x=2, dim_z=1)

    line_filter[i].x = np.array([[2.],
                    [0.]])       # initial state (location and velocity)

    line_filter[i].F = np.array([[1.,1.],
                    [0.,1.]])    # state transition matrix

    line_filter[i].H = np.array([[1.,0.]])    # Measurement function
    line_filter[i].P *= 1000.                 # covariance matrix
    line_filter[i].R = 5                      # state uncertainty
    dt = 0.1
    line_filter[i].Q = Q_discrete_white_noise(2, dt, .1) # process uncertainty


#cap = cv2.VideoCapture(0) #Iš kameros
filename = 'DJI_0295.MP4'
cap = cv2.VideoCapture('dataset debesys/' + filename)
#cap = cv2.VideoCapture('/mnt/48ECBA7C72F0B029/Clouds Video/GH015798.MP4') #Iš video
#cap = cv2.VideoCapture("/home/ivan/Desktop/Straipsniai/Test data/ships and water/Pexels Videos 2257010.mp4") #Iš video
#cap = cv2.VideoCapture("/home/ivan/Desktop/Straipsniai/Test data/Road CAr Pexels Videos 1394254.mp4") #Iš video


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width)
print(frame_height)

writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
writer_edges = skvideo.io.FFmpegWriter("outputvideo_edges.mp4")



devider = frame_width/frame_size


r_frame_width = int(frame_width/devider)
r_frame_height = int(frame_height/devider)
print(r_frame_width)
print(r_frame_height)
image_counter = 0

while(1):

    _, frame = cap.read()
    
    if(frame is not None and frame_counter%frame_skip == 0):
                    
        frame = cv2.resize(frame, (r_frame_width, r_frame_height))


        image_median = np.median(frame)
        frame_detect = frame.copy()
        frame_line = []
        horizon_y = 0

        if horizon_on:
            for y in range(0, frame.shape[0], int(frame.shape[0]/10)):
                r = 0
                g = 0
                b = 0
                for line_pixel in frame[y]:
                    r = r + line_pixel[2]
                    g = g + line_pixel[1]
                    b = b + line_pixel[0]
                r = r / r_frame_width
                g = g / r_frame_width
                b = b / r_frame_width

                if horizon_detect(r, g ,b):
                    #horizon_y = y-horizon_offset
                    horizon_y= r_frame_height-(1)*r_frame_height/parts_count
                    cv2.line(frame_detect, (0, y-horizon_offset), (r_frame_width, y-horizon_offset), (0, 255, 0), 1)
                    break
        else:
            horizon_y= r_frame_height-(1)*r_frame_height/parts_count
            
        if horizon_y == 0:
            horizon_y = r_frame_height
            
        
        #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        lower = 150
        upper = 300
        edges = cv2.Canny(frame_detect, lower, upper)

        lines = []
        lines = cv2.HoughLinesP(edges,rho = 100,theta = 90*np.pi/180,threshold = 10,minLineLength = int(10),maxLineGap = int(5))

        # Apply Hough transform
        detected_circles = cv2.HoughCircles(edges,  
                       cv2.HOUGH_GRADIENT, 7, 12, param1 = 10, 
                       param2 = 20, minRadius = int(5), maxRadius = int(20))
        
        rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

                            
        if ((detected_circles is None)!= True or (lines is None) != True) and detect_on and line_detection_on:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 >= frame.shape[1]:
                    x1 = frame.shape[1]-1
                if x2 >= frame.shape[1]:
                    x2 = frame.shape[1]-1
                if y1 >= frame.shape[0]:
                    y1 = frame.shape[0]-1
                if y2 >= frame.shape[0]:
                    y2 = frame.shape[0]-1
                    
                b = frame[y1][x1][0]
                g = frame[y1][x1][1]
                r = frame[y1][x1][2]                       
    
                if y1 < horizon_y and y2 < horizon_y and cloud_detect(r, g, b):
                    cv2.line(frame_detect, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.line(rgb_edges, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                    
            for circle in detected_circles[0]:
                x, y, ra = circle
                x = int(x)
                if x >= frame.shape[1]:
                    x = frame.shape[1] - 1
                y = int(y)
                if y >= frame.shape[0]:
                    y = frame.shape[0] - 1  
                b = frame[y][x][0]
                g = frame[y][x][1]
                r = frame[y][x][2]
                if y < horizon_y and cloud_detect(r, g, b):
                    cv2.circle(frame_detect, (x, y), ra, (255, 255, 0), 1)
                    cv2.circle(rgb_edges, (x, y), ra, (255, 255, 0), 1)
                    
            if grid_on:
                conn = sqlite3.connect('clouds.db')
                c = conn.cursor()
                kalman_counter = 0
                for i in range(parts_count):
                    x = int(i*r_frame_width/parts_count)
                    for z in range(parts_count):
                        y = int(z*r_frame_height/parts_count)
                        if  y < horizon_y:
                            color = (50, 50, 150)
                            w_ofset = (x + int(r_frame_width/parts_count))
                            h_ofset = (y + int(r_frame_height/parts_count))
                             
                            cv2.rectangle(frame_detect, (x, y), (x + w_ofset, y + h_ofset), color, 1)
                            cv2.rectangle(rgb_edges, (x, y), (x + w_ofset, y + h_ofset), color, 1)
                            cicle_count_in_square = 0
                            if (detected_circles is None) != True:
                                for circle in detected_circles[0]:
                                    xr, yr, ra = circle
                                    
                                    xr = int(xr)
                                    if xr >= frame.shape[1]:
                                        xr = frame.shape[1] - 1
                                    yr = int(yr)
                                    if yr >= frame.shape[0]:
                                        yr = frame.shape[0] - 1
                            
                                    b = frame[yr][xr][0]
                                    g = frame[yr][xr][1]
                                    r = frame[yr][xr][2]
                                    if yr < horizon_y and cloud_detect(r, g, b) and xr + ra > x and xr + ra < x + int(r_frame_width/parts_count)  and yr + ra > y and yr + ra < y + int(r_frame_height/parts_count):
                                        cicle_count_in_square += 1
                            
                                font_face = cv2.FONT_HERSHEY_SIMPLEX
                                scale = 0.4
                                         
                                circle_filter[kalman_counter].predict()
                                circle_filter[kalman_counter].update(cicle_count_in_square)

                                line_out = line_filter[kalman_counter].x[0][0]
                                circle_out = circle_filter[kalman_counter].x[0][0]

                                if circle_out + line_out > detection_edge:
                                    color = (0, 255, 255)
                                else:
                                    color = (255, 0, 255)
                                     
                                cv2.putText(frame_detect, str(round(circle_out, 2)), (x,y+60), font_face, scale, color, 1, cv2.LINE_AA)
                                cv2.putText(rgb_edges, str(round(circle_out, 2)), (x,y+60), font_face, scale, color, 1, cv2.LINE_AA)
                            
                            line_count_in_square = 0
                            if (lines is None) != True:
                                for line in lines:
                                    x1, y1, x2, y2 = line[0]

                                    if x1 >= frame.shape[1]:
                                        x1 = frame.shape[1]-1
                                    if x2 >= frame.shape[1]:
                                        x2 = frame.shape[1]-1
                                    if y1 >= frame.shape[0]:
                                        y1 = frame.shape[0]-1
                                    if y2 >= frame.shape[0]:
                                        y2 = frame.shape[0]-1
                                        
                                    b = frame[y1][x1][0]
                                    g = frame[y1][x1][1]
                                    r = frame[y1][x1][2]
                                    
                                    if(y1 < horizon_y and cloud_detect(r, g, b) and y2 < horizon_y and x1 > x and x2 < x + int(r_frame_width/parts_count)  and y1 > y and y2 < y + int(r_frame_height/parts_count)):
                                        line_count_in_square += 1
                                font_face = cv2.FONT_HERSHEY_SIMPLEX
                                scale = 0.4


                                line_filter[kalman_counter].predict()
                                line_filter[kalman_counter].update(line_count_in_square)

                                line_out = line_filter[kalman_counter].x[0][0]
                                circle_out = circle_filter[kalman_counter].x[0][0]

                                if circle_out + line_out > detection_edge:
                                    color = (0, 255, 0)
                                else:
                                    color = (0, 0, 255)
                                    
                                cv2.putText(frame_detect, str(round(line_out, 2)), (x,y+30), font_face, scale, color, 1, cv2.LINE_AA)
                                cv2.putText(rgb_edges, str(round(line_out, 2)), (x,y+30), font_face, scale, color, 1, cv2.LINE_AA)

                            croped_visual_img = frame[y:h_ofset, x:w_ofset]
                            #cv2.imshow('croped',croped_visual_img)
                            if frame_counter%write_counter == 0:
                                cv2.imwrite('crop_image/' + filename + str(image_counter) +'x'+ str(x) +'y' + str(y) + '.jpg', croped_visual_img)
                                orig_image_name =  filename + str(image_counter) + '.jpg'
                                image_cropped_name = filename + str(image_counter) +'x'+ str(x) +'y' + str(y) + '.jpg'
                                x_coord = x
                                y_coord = y
                                lines_count = line_out
                                circles_count = circle_out
                                is_cumulus = 0
                                

                                c.execute("INSERT INTO cumulus_expert(filename, image_cropped_name, x_coord, y_coord, lines_count, circles_count, is_cumulus) VALUES('" + orig_image_name + "','" + image_cropped_name  + "', '" + str(x_coord) +"', '" + str(y_coord)+"', '" + str(lines_count) +"', '" + str(circles_count) +"', '" + str(is_cumulus) +"');")


                        kalman_counter +=1

        conn.commit()
        conn.close()             
        cv2.imshow('Original',frame_detect) 
        cv2.imshow('Edges7878',rgb_edges)

        frame_detect = cv2.cvtColor(frame_detect, cv2.COLOR_RGB2BGR)
        rgb_edges = cv2.cvtColor(rgb_edges, cv2.COLOR_RGB2BGR)
        vis = np.concatenate((frame_detect, rgb_edges), axis=1)
#        cv2.imshow('vis',vis)
        writer.writeFrame(frame_detect)
        writer_edges.writeFrame(rgb_edges)

        frame_detect = cv2.cvtColor(frame_detect,cv2.COLOR_RGB2BGR)
        if frame_counter%write_counter == 0:
            cv2.imwrite('orig_image/' + filename + str(image_counter) + '.jpg', frame_detect)
        image_counter += 1
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        cap.grab()
    frame_counter += 1
#writer.close()
cv2.destroyAllWindows()
cap.release()
