# Document-Scanner-Project

step 1 -Resize the Image and Convert to GrayScale

step 2 -Apply Guassian Blur to smoothen the image.

step 3 -Use Canny Edge detection to detect all the edges

step 4 -Find the boundary of the page using contours

step 5 -Extract the boundary and Map it to a new height*width windows (For find width and height use distance formula)

step 6 -Apply Perspective transform for mapping, this gives the top view or bird eye view with scanned effect.

As I mention in code the Perspective transform is point sensitive so for finding the exact order of points
we need to identify the top left, top right, and bottom right, bottom left points.

for example->   (x,y) , (x+w,y) , (x,y+h) , (x+w,y+h) 
    
       
       
    If we add the x and y coordinate of each points and find minimun amd maximun sum coordinates then 
    *  minimum sum coordinate is top left point 
    *  maximum sum coordinate is bottom right point 
    
    
    
     If we subtract the x and y coordinate of each points and find minimun amd maximun sum coordinates then 
    *  minimum diff coordinate is top right point 
    *  maximum diff coordinate is bottom left point 
    
    Once we get our points we apply warp perspective and get a scanned image.
    
    ************       Thanks for reading      *************
    
    
