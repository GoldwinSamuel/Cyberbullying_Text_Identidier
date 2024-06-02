// window.onload = function() {
//     var viewportValuesDiv = document.getElementById("viewportValues");

//     // Function to update viewport values
//     function updateViewportValues() {
//         var width = window.innerWidth;
//         var height = window.innerHeight;
//         viewportValuesDiv.textContent = "Viewport Width: " + width + "px, Viewport Height: " + height + "px";
//     }

//     // Call updateViewportValues() whenever window is resized
//     window.onresize = updateViewportValues;

//     // Initial call to updateViewportValues()
//     updateViewportValues();
// };
// window.onload = function() {
//     var object = document.getElementById("predict_btn");

//     // Function to adjust object position based on viewport size
//     function adjustObjectPosition() {
//         var viewportWidth = window.innerWidth;
//         var viewportHeight = window.innerHeight;
//         var desiredRatio = 2; // Aspect ratio of the object

//         // Calculate the width and height based on the desired aspect ratio
//         var height = object.clientHeight;
//         var width = height * desiredRatio;

//         // Calculate the left and top positions to maintain the aspect ratio
//         var left = (viewportWidth - width) / 2;
//         var top = (viewportHeight - height) / 2;s

//         // Set the left and top positions of the object
//         object.style.left = left + "px";
//         object.style.top = top + "px";
//     }

//     // Call adjustObjectPosition() whenever window is resized
//     window.onresize = adjustObjectPosition;

//     // Initial call to adjustObjectPosition()
//     adjustObjectPosition();
// };
// width 744 font size changes