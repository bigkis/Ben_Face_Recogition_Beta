package app.com.example.android.ben_face_recogition_beta;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.util.Vector;


public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final String TAG = "Ben Debugs";
    String ORL = "/storage/emulated/0/BensProject/ORLDatabase/";
    String DisplayTest = "For now no button has been pressed";

    ImageView iv_taken;
    ImageView iv_mapback;
    long mapback_subject;
    double mapback_image;
    Bitmap myBitmap;

    int Subjects_Training_Min = 1,Subjects_Training_Max = 41, Images_Training_Min = 1, Images_Training_Max = 6;
    int Subjects_Input_Min = 1,Subjects_Input_Max = 41 , Images_Input_Min = 6, Images_Input_Max = 11;

    private CameraBridgeViewBase mOpenCvCameraView;
    boolean touched = false;

    public class Ben extends BaseLoaderCallback {
        public Ben(Context AppContext)
        {
            super(AppContext);


        }

        public Mat cropped;
        public Mat Images_Training;
        public Mat Images_Training_Mean;
        public Mat EigenVectors;
        public Mat Coefficients_Training;
        public Mat Who;

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");


                    cropped  = new Mat();
                    Images_Training = new Mat();
                    Images_Training_Mean = new Mat();
                    EigenVectors = new Mat();
                    Coefficients_Training = new Mat();
                    Who = new Mat();

                    Images_Training = CreateImageMatrix(Subjects_Training_Min,Subjects_Training_Max,Images_Training_Min,Images_Training_Max);
                    Images_Training_Mean = CalculateMean(Images_Training);
                    Mat Images_Normalized_Training = SubstractMatrices(Images_Training, Images_Training_Mean);
                    Mat Images_Training_T = TransposeMatrix(Images_Training);
                    EigenVectors = CreateEigenVectors(Images_Training_T);
                    Coefficients_Training = CalculateCoefficients(Images_Normalized_Training, EigenVectors);


                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    }

    private Ben mLoaderCallback=null;



    public Mat CreateImageMatrix(int Subjects_Training_Min, int Subjects_Training_Max, int Images_Training_Min, int Images_Training_Max){

        Vector trainingImages = new Vector();

        for(int a = Subjects_Training_Min; a < Subjects_Training_Max; a++){
            for(int b = Images_Training_Min; b < Images_Training_Max; b++){

                Mat image = new Mat();
                image =Highgui.imread(ORL + "s" + a + "/" + b + ".png", 0);
                if(image.empty()){
                    image =Highgui.imread(ORL + "s" + a + "/" + b + ".pgm", 0);
                }
                trainingImages.add(image);
            }
        }



        Mat firstTrainingImage = (Mat) trainingImages.get(0); //get first image
        int numberOfCells = firstTrainingImage.rows() * firstTrainingImage.cols();

        Mat phiMatrix = new Mat(numberOfCells, trainingImages.size(), CvType.CV_32FC1);
        for (int i = 0; i < trainingImages.size(); i++) {
            Mat phiTrainingImageColumn = phiMatrix.col(i);
            Mat trainingImage = (Mat) trainingImages.get(i);
            Mat reshaped = trainingImage.reshape(1, numberOfCells);
            reshaped.convertTo(phiTrainingImageColumn, CvType.CV_32FC1);
        }
        return(phiMatrix);

    }

    public Mat CalculateMean(Mat phiMatrix){

        Mat mean = Mat.zeros(phiMatrix.rows(),1, CvType.CV_32FC1);

        for(int i =0; i<phiMatrix.cols(); i++){
            Core.add(mean.col(0), phiMatrix.col(i), mean.col(0));
        }


        for(int a=0;a<mean.rows();a++){
            for(int b=0;b<mean.cols();b++) {
                double[] val = mean.get(a,b);
                val[0]/=phiMatrix.cols();
                mean.put(a,b,val[0]);
            }
        }
        return(mean);

    }

    public Mat TransposeMatrix(Mat images){
        Mat matT = new Mat();
        Core.transpose(images, matT);
        return(matT);

    }

    public Mat SubstractMatrices(Mat images, Mat mean){
        Mat A = new Mat(images.rows(), images.cols(), CvType.CV_32FC1);
        for(int i =0; i<images.cols(); i++){
            Core.subtract(images.col(i),mean,A.col(i));
        }
        return A;
    }

    public Mat CreateEigenVectors(Mat images){

        Mat eigenVectors = new Mat();
        Core.PCACompute(images, new Mat(), eigenVectors);
        return eigenVectors;

    }

    public Mat CalculateCoefficients(Mat A, Mat eigenVectors){
        Mat weights = new Mat(eigenVectors.rows(),A.cols(),CvType.CV_32FC1);
        for(int i=0; i<A.cols();i++){
            Core.gemm(eigenVectors, A.col(i), 1, new Mat() , 0 , weights.col(i) , 0);
        }
        return(weights);
    }

    public Mat CompareArray (Mat Coefficients_Training, Mat Coefficients_Input){

        Mat Result = Mat.zeros(2,Coefficients_Training.cols(),CvType.CV_32FC1);
        double temp = 0;
        for(int j=0; j<Coefficients_Input.cols() ;j++) {

            Mat Decision = Mat.zeros(1, Coefficients_Training.cols(), CvType.CV_32FC1);

            for (int i = 0; i < Coefficients_Training.cols(); i++) {
                temp =Core.norm(Coefficients_Training.col(i),Coefficients_Input.col(j), Core.NORM_L2);
                Decision.put(0,i,temp);
            }


            Core.MinMaxLocResult mmr = Core.minMaxLoc(Decision);
            double xLoc = mmr.minLoc.x;
            double yLoc = mmr.minLoc.y;
            Result.put(0,j,xLoc);
            Result.put(1,j,yLoc);
        }

        return Result;

    }

    public Mat Identify(Mat Result, int Subjects_Training_Min, int Subjects_Training_Max, int Images_Training_Min, int Images_Training_Max){

        Mat FinalResult = Mat.zeros(2, Result.cols(), CvType.CV_32FC1);

        int Number_of_Training_Subjects = Subjects_Training_Max - Subjects_Training_Min;
        int Number_of_Training_Images = Images_Training_Max - Images_Training_Min;


        int k = 0;
        for (int j = 0; j < Number_of_Training_Subjects; j++) {
            for (int i = 0; i < Number_of_Training_Images; i++) {

                double[] image = Result.get(0, k);
                Double d = new Double(image[0]);
                int image1 = d.intValue();
                int count = 0;
                image1 = (image1 + Number_of_Training_Images ) / Number_of_Training_Images ;
                FinalResult.put(0, k, image1);


                if (j + 1 == image1) {
                    count = 1;
                } else {
                    count = 0;
                }
                FinalResult.put(1, k, count);
                k++;


            }

        } return FinalResult;
    }


    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        iv_taken = (ImageView)findViewById(R.id.imageView_taken);
        iv_mapback = (ImageView)findViewById(R.id.imageView_mapback);

        iv_mapback.setImageResource(R.drawable.ic_launcher);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        Button btn = (Button)findViewById(R.id.button);
        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                touched = true;
                Log.i(TAG,"THE FUCKING BUTTON WAS CLICKED");
                Log.i(TAG,"THE VALUE OF TOUCHED IS = " + touched);
            }
        });

        this.mLoaderCallback=new Ben(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        Mat rgba = inputFrame.rgba();




        if(touched){
            Log.i(TAG,"touched is = to " + touched);

            Mat gray = inputFrame.gray();

            //Cropping
            Rect roi = new Rect(500,300,368,448);
            Mat cropped = new Mat(gray,roi);

            Mat croppedclone = cropped.clone();

            //Correct Type
            Mat correcttype = new Mat();
            croppedclone.assignTo(correcttype,CvType.CV_32FC1);

            Mat croppedcloneT = new Mat(112,92,CvType.CV_32FC1);
            Core.transpose(croppedclone, croppedcloneT);



            Size newSize = new Size(92,112);
            Mat resized = new Mat();


            Imgproc.resize(correcttype, resized, newSize);
            Core.transpose(resized, resized);

            Mat reshaped = resized.reshape(1, 10304);

            Mat Images_Normalized_Input = SubstractMatrices(reshaped,mLoaderCallback.Images_Training_Mean );
            Mat Coefficients_Input = CalculateCoefficients(Images_Normalized_Input, mLoaderCallback.EigenVectors);
            Mat Result = CompareArray(mLoaderCallback.Coefficients_Training, Coefficients_Input);
            Mat Who = Identify(Result, 1, 2, 1, 2);

            double[] AlmostThere = Who.get(0,0);
            Double gotit = new Double(AlmostThere[0]);
            DisplayTest = gotit.toString();


            double fPart;
            double numsubject;
            double s1;
            double s2;
            double s3;



            s1 = gotit;
            s2 = s1 -1;
            s3 = s2/5;
            numsubject = s3 +1;
            mapback_subject = (long)numsubject;
            fPart = numsubject - mapback_subject;


            //0
            if (fPart < 0.1){
                mapback_image = 1;
            }
            //0.2
            if (fPart>0.1 && fPart<0.3){
                mapback_image = 2;
            }
            //0.4
            if (fPart>0.3 && fPart<0.5){
                mapback_image = 3;
            }
            if (fPart>0.5 && fPart<0.7){
                mapback_image = 4;
            }
            if (fPart>0.5 && fPart<0.9){
                mapback_image = 5;
            }

            DisplayTest = "Subject = " + String.valueOf(mapback_subject) + "       Image = " + String.valueOf(mapback_image);

/*            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            Bitmap bitmap = BitmapFactory.decodeFile(ORL + "s" + String.valueOf(mapback_subject) + "/" + String.valueOf((int)mapback_image) + ".pgm",options);
            iv_mapback.setImageBitmap(bitmap);*/


/*            File imFile = new File(ORL + "s" + String.valueOf(mapback_subject) + "/" + String.valueOf((int)mapback_image) + ".pgm");
            if(imFile.exists()) {
                myBitmap = BitmapFactory.decodeFile(imFile.getAbsolutePath());
                iv_mapback.setImageBitmap(myBitmap);

            */




            Log.i(TAG,"Picture Saved");

            touched = false;





        }

        Core.rectangle(rgba, new Point(500, 300), new Point(768, 748), new Scalar(0, 255, 255));
        Core.putText(rgba, DisplayTest, new Point(100, 500), 3, 1, new Scalar(255, 0, 0, 255), 2);

        return rgba;
    }

}