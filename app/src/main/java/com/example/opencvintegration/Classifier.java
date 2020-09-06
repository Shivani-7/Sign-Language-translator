package com.example.opencvintegration;
import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.widget.ArrayAdapter;

import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import androidx.annotation.NonNull;

class Classifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3;
    private int IMAGE_MEAN=0;
    private float IMAGE_STD=255.0f;
    private float MAX_RESULTS=3;
    private float THRESHOLD=0.4f;
    Classifier(AssetManager assetManager,String modelPath, String labelPath,int inputSize) throws Exception {
        INPUT_SIZE=inputSize;
        Interpreter.Options options=new Interpreter.Options();
        options.setNumThreads(5);
        options.setUseNNAPI(true);
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        labelList=loadLabelList(assetManager,labelPath);
    }
    class Recognition{
        String id="";
        String title="";
        float confidence=0f;
        public Recognition(String i,String s,float confidence)
        {
            id=i;
            title=s;
            this.confidence=confidence;
        }

        @NonNull
        @Override
        public String toString() {
            return "title = "+title+",Confidence = "+confidence;
        }
    }
    private MappedByteBuffer loadModelFile(AssetManager assetManager,String MODEL_FILE) throws IOException{
        AssetFileDescriptor fileDescriptor=assetManager.openFd(MODEL_FILE);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws Exception{
        List<String> labelList=new ArrayList<>();
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while((line=reader.readLine())!=null)
        {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }
    List<Recognition> recognizeImage(Bitmap bitmap)
    {
        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);
        float [][]result =new float[1][labelList.size()];
        interpreter.run(byteBuffer,result);
        return getSortedResultfloat(result);
    }
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap)
    {
        ByteBuffer byteBuffer=ByteBuffer.allocateDirect(4*INPUT_SIZE*INPUT_SIZE*PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[INPUT_SIZE*INPUT_SIZE];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;
        for(int i=0;i<INPUT_SIZE;++i)
        {
            for(int j=0;j<INPUT_SIZE;++j)
            {
                final int val=intValues[pixel++];
                byteBuffer.putFloat((((val>>16)& 0xff)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val>>0)& 0xff)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val>>0xff)& 0xff)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }
    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResultfloat(float[][] labelProbArray){
        PriorityQueue<Recognition> pq=new PriorityQueue<>(
                (int) MAX_RESULTS,
                new Comparator<Recognition>() {
                    @Override
                    public int compare(Recognition o1, Recognition o2) {
                        return Float.compare(o1.confidence,o2.confidence);
                    }
                }
        );
        for(int i=0;i<labelList.size();++i)
        {
            float confidence=labelProbArray[0][i];
            if(confidence>THRESHOLD)
            {
                pq.add(new Recognition(""+i,labelList.size()>i?labelList.get(i):"unknown",confidence));
            }
        }

        final ArrayList<Recognition> recognitions=new ArrayList<>();
        int recognitionSize=(int) Math.min(pq.size(),MAX_RESULTS);
        for(int i=0;i<recognitionSize;++i)
        {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }
}


