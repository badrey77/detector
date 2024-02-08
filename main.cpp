#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace cv;
using namespace cv::face;
using namespace std;

int main() {
    // Load face detection and LBPH face recognizer
    CascadeClassifier face_cascade;
    face_cascade.load(R"(C:\msys64\clang64\share\opencv4\haarcascades\haarcascade_frontalface_default.xml)");

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();

    // Load pre-trained model for face recognition
    model->read("lbph_model.yml");

    // Capture video from webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open webcam" << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 4);

        // Iterate through detected faces
        for (const Rect& face : faces) {
            // Draw rectangle around the face
            rectangle(frame, face, Scalar(255, 0, 0), 2);

            // Extract face region
            Mat face_roi = gray(face);

            // Perform face recognition
            int label = -1;
            double confidence = 0.0;
            model->predict(face_roi, label, confidence);

            // Display recognized label and confidence
            string label_text = format("Person: %d Confidence: %.2f", label, confidence);
            Point text_pos(face.x, face.y - 10);
            putText(frame, label_text, text_pos, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
        }

        // Display the frame with face rectangles and recognized labels
        imshow("Face Recognition", frame);

        // Break loop if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
