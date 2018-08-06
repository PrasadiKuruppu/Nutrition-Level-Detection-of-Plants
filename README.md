# Nutrition-Level-Detection-of-Plants

Project Description: 

	Nitrogen(N),Phosphorus(P) and Potassium(K) are the primary nutrients 
for the plants. In this code I have used color of the leaf to detect the type of low 
nutrition level.Nitrogen and Potassium have yellow color and Phosphorus has 
a purple color when the leaves are suffering from low nutrition level 
of the respective nutrients.

	I have provided all the requiered details of the code in each line of the code. 
In the code first I have identified the difference between yellow and purple 
color using HSV color model. If the leaf is having purple color then the type will 
get identified as Phosphorus. I have used percentage of yellow colort part of the 
leaf to identify the difference between the types Nitrogen and Potassium. Because 
for low Potassium level the edges of the leaf get the yellow color and then spread 
to the middle areas of the leaf. But for low Nitrogen level whole leaf become into 
yellow color.

About Inculded Files:

	1.NUTRITION_CODE
	
	2.NUTRITION_GUI
	
	3.Images
	
		(i)  Low Nitrogen Level Leaf Images: N 1,2
		(ii) Low Potassium Level Leaf Images: K 1,2,3
		(iii)Low Phosphorus Level Leaf Images: P 1,2,3,4,5
		(iv) Healthy Leaves: H 1,2,3,4,5,6  
		(v)  Background for Python GUI: GUI_Background

Note: Make sure you keep all the images in the same folder which you save the code. 
      Otherwise you have to give extensions for the images explicitly in the respective 
      lines of the code.

Test the Code:

	Input--> image(Close Images are requied. Otherwise all leaves will get deleted 
			when we apply grabcut in the code)
	Output--> 1.Average of Hue value
          	  2.Percentage of the leaf having low nutrition level(different color part)
          	  3.Type of low nutrition(Nitrogen,Potassium,Phosphorus)
          	  4.Time spent to run the code

	* If the input is a healthy leaf the output will say it is healthy.
	* In H6 image you will get an error message because the image is not focused 
	  on any specific leaf.
	* All the comments are provided in the code.
