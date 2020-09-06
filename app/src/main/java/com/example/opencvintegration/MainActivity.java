package com.example.opencvintegration;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    CameraBridgeViewBase cameraBridgeViewBase;
    Mat mat1,mat2,mat3;
    BaseLoaderCallback mLoaderCallback;


    private  int mInputsize=50;
    private String mModelPath="model.tflite";
    private String mLabelPath="label.txt";
    protected Interpreter tflite;
    private Classifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraBridgeViewBase=(JavaCameraView)findViewById(R.id.myCameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
          mLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS: {
                        Log.i("", "OpenCV loaded successfully");
                        cameraBridgeViewBase.enableView();
                        break;
                    }
                    default: {
                        super.onManagerConnected(status);
                        break;
                    }
                }
            }
        };

          try {
              initClassifier();
          }catch (Exception e)
          {
              e.printStackTrace();
          }
    }



    private  void initClassifier() throws Exception {
        classifier=new Classifier(getAssets(),mModelPath,mLabelPath,mInputsize);
}

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mat1=inputFrame.rgba();
        Size sz=new Size(400,400);
        Imgproc.rectangle(mat1,new Point(600,600), new Point(100, 100),
                new Scalar(0, 255, 0),2);
        /*Rect extractedRect= new Rect(new Point(600,600), new Point(100, 100));
        Mat croppedimg=new Mat(mat1,extractedRect);
        Mat pmat=new Mat();
        Imgproc.cvtColor(croppedimg, pmat,Imgproc.COLOR_BGR2GRAY);
        Mat thres=new Mat();
        Imgproc.adaptiveThreshold(pmat,thres,125,Imgproc.THRESH_BINARY_INV,Imgproc.THRESH_OTSU,210,255);*/
        Bitmap bmp=null;
        bmp=Bitmap.createBitmap(mat1.cols(),mat1.rows(),Bitmap.Config.ARGB_8888);
        List<Classifier.Recognition> result=classifier.recognizeImage(bmp);
        Toast.makeText(this, result.get(0).toString(), Toast.LENGTH_SHORT).show();
        return mat1;
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mat1=new Mat(width,height, CvType.CV_8UC4);
        mat2=new Mat(width,height, CvType.CV_8UC4);
        mat3=new Mat(width,height, CvType.CV_8UC4);

    }

    @Override
    public void onCameraViewStopped() {
        mat1.release();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null)
        {
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug())
        {
            Toast.makeText(getApplicationContext(),"There is problem in opencv",Toast.LENGTH_SHORT).show();
        }
        else
        {
            mLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase!=null)
        {
            cameraBridgeViewBase.disableView();
        }
    }
}
