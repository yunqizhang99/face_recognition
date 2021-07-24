/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelUuid;
import android.os.Parcelable;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.RequiresApi;

import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.tensorflow.lite.examples.detection.ble5performacetest.Assembler;
import org.tensorflow.lite.examples.detection.ble5performacetest.BeaconBroadcast;
import org.tensorflow.lite.examples.detection.ble5performacetest.Fragmenter;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();


  // FaceNet
//  private static final int TF_OD_API_INPUT_SIZE = 160;
//  private static final boolean TF_OD_API_IS_QUANTIZED = false;
//  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
//  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";

  // MobileFaceNet
  private static final int TF_OD_API_INPUT_SIZE = 112;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";


  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  //private static final int CROP_SIZE = 320;
  //private static final Size CROP_SIZE = new Size(320, 320);


  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  private static int OUTPUT_SIZE = 10000;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private SimilarityClassifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;
  private boolean addPending = false;
  //private boolean adding = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  //private Matrix cropToPortraitTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // Face detector
  private FaceDetector faceDetector;

  // here the preview image is drawn in portrait way
  private Bitmap portraitBmp = null;
  // here the face is cropped and drawn
  private Bitmap faceBmp = null;

  private FloatingActionButton fabAdd;

  //private HashMap<String, Classifier.Recognition> knownFaces = new HashMap<>();

  private BluetoothManager manager;
  private BluetoothAdapter btAdapter;
  private BluetoothLeScanner bluetoothLeScanner;
  private ScanSettings settings;
  private List<ScanFilter> filters;
  private static final int REQUEST_ENABLE_BLUETOOTH = 1;
  private static final ParcelUuid SERVICE_UUID =
          ParcelUuid.fromString("0000FEAA-0000-1000-8000-00805F9B34FB");
  private ArrayList<String> names = new ArrayList<String>();
  private static final int SEPARATOR_CHAR = 42;
  private HashMap<String,String> currentRec = new HashMap<String,String>();

  @RequiresApi(api = Build.VERSION_CODES.O)
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    initializeBt();

    fabAdd = findViewById(R.id.fab_add);
    fabAdd.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        onAddClick();
      }
    });

    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .build();


    FaceDetector detector = FaceDetection.getClient(options);

    faceDetector = detector;


    Log.v("here", "here5");
    scanLeDevice(true);
    //checkWritePermission();

  }



  private void onAddClick() {

    addPending = true;
    //Toast.makeText(this, "click", Toast.LENGTH_LONG ).show();

  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);


    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      //cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


    int targetW, targetH;
    if (sensorOrientation == 90 || sensorOrientation == 270) {
      targetH = previewWidth;
      targetW = previewHeight;
    }
    else {
      targetW = previewWidth;
      targetH = previewHeight;
    }
    int cropW = (int) (targetW / 2.0);
    int cropH = (int) (targetH / 2.0);

    croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

    portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
    faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropW, cropH,
                    sensorOrientation, MAINTAIN_ASPECT);

//    frameToCropTransform =
//            ImageUtils.getTransformationMatrix(
//                    previewWidth, previewHeight,
//                    previewWidth, previewHeight,
//                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);


    Matrix frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    targetW, targetH,
                    sensorOrientation, MAINTAIN_ASPECT);



    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }


  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
    faceDetector
            .process(image)
            .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
              @Override
              public void onSuccess(List<Face> faces) {
                if (faces.size() == 0) {
                  updateResults(currTimestamp, new LinkedList<>());
                  return;
                }
                runInBackground(
                        new Runnable() {
                          @Override
                          public void run() {
                            onFacesDetected(currTimestamp, faces, addPending);
                            addPending = false;
                          }
                        });
              }

            });


  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }


  // Face Processing
  private Matrix createTransform(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation) {

    Matrix matrix = new Matrix();
    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

    if (applyRotation != 0) {

      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;

  }

  public void scanLeDevice(final boolean enable) {
    if (enable) {
      bluetoothLeScanner.startScan(filters, settings, leScanCallback);
    } else {
      bluetoothLeScanner.stopScan(leScanCallback);
    }
  }

  private ScanCallback leScanCallback = new ScanCallback() {
    @Override
    public void onScanResult(int callbackType, ScanResult result) {
      Log.v("here", "here6");
      Context context = getApplicationContext();
      CharSequence text = "BLE Receiving: LE ScanCallBack!";
      int duration = Toast.LENGTH_SHORT;

      Toast toast = Toast.makeText(context, text, duration);
      toast.show();

      Log.i("callbackType", String.valueOf(callbackType));
      //Log.i("result", result.getDataStatus()+"");

//            if (result.getDataStatus() == 0) {
//                complete++;
//            } if (result.getDataStatus() == 2) {
//                truncated++;
//            }
//            Log.i("collision rate: ", truncated/((complete+truncated)*1.00)+"");

      Log.v("here", "here22 " + result.getScanRecord().getDeviceName());
      Log.v("here", "here23 "+result.getScanRecord().getBytes().length);
      String address = result.getDevice().getName();
      //byte[] pData = Assembler.gather(address, result.getScanRecord().getServiceData(SERVICE_UUID));
      byte [] pData = Assembler.gather(address, result.getScanRecord().getBytes());
      //Log.v("here", "here21 "+ pData.length);
      if (pData != null) {
        Log.v("here", "here2");
        update(pData);
        Log.v("here", "here3");
      }
    }

    @Override
    public void onBatchScanResults(List<ScanResult> results) {
      for (ScanResult sr : results) {
        Log.i("ScanResult - Results", sr.toString());
      }
    }

    @Override
    public void onScanFailed(int errorCode) {
      Log.e("Scan Failed", "Error Code: " + errorCode);
    }
  };

  private void update(byte[] data){
    Log.v("DetectorActivity 488", "Debug Okay1" + data);
    int index = 0;
    int nextStart = 0;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    Log.v("DetectorActivity 488", "Debug Okay2");
    byte[] nameByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    Log.v("DetectorActivity 488", "Debug Okay3");
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] idByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] titleByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] distByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] locLByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] locTByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] locRByte = Arrays.copyOfRange(data, nextStart, index);
    nextStart = index;
    while (data[index] != (byte)SEPARATOR_CHAR){
      Log.v("DetectorActivity 488", "Debug Okay4");
      index++;
    }
    byte[] locBByte = Arrays.copyOfRange(data, nextStart, index);
    byte[] cropByte = Arrays.copyOfRange(data, index, data.length);

    Log.v("DetectorActivity 488", "Debug Exception14");

    String name = "";
    String id = "";
    String title = "";
    Float dist = 0.0f;
    Float locL = 0.0f;
    Float locT = 0.0f;
    Float locR = 0.0f;
    Float locB = 0.0f;
    RectF loc;
    Bitmap crop;
//    byte[] name = Arrays.copyOfRange(data,0,index);
//    index++;
//    int recStart = index;
//    while (data[index] != (byte)SEPARATOR_CHAR){
//      index++;
//    }
//    byte[] recData = Arrays.copyOfRange(data,recStart,index);
//    index++;
//    byte[] pantsData = Arrays.copyOfRange(data,index,data.length);
//    String nameStr = new String(name);
//    String recStr = new String(recData);
//    String pantsStr = new String(pantsData);

//    SimilarityClassifier.Recognition receivedRec = null;
//    ByteArrayInputStream bis = new ByteArrayInputStream(recData);
//    ObjectInput in = null;

    //Map.Entry<String, String> map = null;
    try {
//      ByteArrayOutputStream out = new ByteArrayOutputStream();
//      ObjectOutputStream objectOutputStream= new ObjectOutputStream(out);
//      objectOutputStream.write(recData);
//      objectOutputStream.flush();
//
//      ObjectInput input = new ObjectInputStream(new ByteArrayInputStream(out.toByteArray()));
//    System.out.println("input.available(): " + input.available());
//    System.out.println("input.readByte(): " + input.readByte());

//      map = (Map.Entry<String, String>) input.readObject();
//      String rec = map.getValue();
//      String[] values = rec.split("#");
//      float f = 0;
//      if (!values[2].equals("")) {
//        f = Float.parseFloat(values[2]);
//      }
//      String[] rectFString = values[3].split("\\(");
//      rectFString = rectFString[1].split("\\)");
//      rectFString = rectFString[0].split(",");
//      float one = Float.parseFloat(rectFString[0]);
//      float two = Float.parseFloat(rectFString[1]);
//      float three = Float.parseFloat(rectFString[2]);
//      float four = Float.parseFloat(rectFString[3]);
//      RectF rectF = new RectF(one, two, three, four);
//      SimilarityClassifier.Recognition recognition = new SimilarityClassifier.Recognition(values[0], values[1], f, rectF);
//      recognition.setExtra(getFloatArrayObject(values[4]));
//      //registered.put(map.getKey(), recognition);
//      detector.register(map.getKey(), recognition);
      name = new String (nameByte);
      id = new String (idByte);
      title = new String (titleByte);

      Log.v("DetectorActivity 488", "Debug Exception13");

      dist = ByteBuffer.wrap(distByte).getFloat();

      Log.v("DetectorActivity 488", "Debug Exception12");

      locL = ByteBuffer.wrap(locLByte).getFloat();
      locT = ByteBuffer.wrap(locTByte).getFloat();
      locR = ByteBuffer.wrap(locRByte).getFloat();
      locB = ByteBuffer.wrap(locBByte).getFloat();

      loc = new RectF(locL, locT, locR, locB);

      Log.v("DetectorActivity 488", "Debug Exception11");

      BitmapFactory.Options options = new BitmapFactory.Options();
      options.inMutable = true;
      crop = BitmapFactory.decodeByteArray(cropByte, 0, cropByte.length, options);
      Canvas canvas = new Canvas(crop);

      Log.v("DetectorActivity 488", "Debug Exception10");

      SimilarityClassifier.Recognition receivedRec = new SimilarityClassifier.Recognition(id, title, dist, loc);
      receivedRec.setCrop(crop);

      detector.register(name, receivedRec);

      Log.v("here11", name);
      Log.v("DetectorActivity 488", "Debug Okay");
    } catch (Exception ex){
      Log.v("DetectorActivity 488", "Debug Exception");
    }

//    finally {
//      try {
//        if (in != null) {
//          in.close();
//        }
//      } catch (IOException ex) {
//        // ignore close exception
//      }
//    }

    Context context = getApplicationContext();
    CharSequence text = "BLE Receiving: "+name;
    int duration = Toast.LENGTH_SHORT;

    Toast toast = Toast.makeText(context, text, duration);
    toast.show();

    // yunqi
  }

  static Object getFloatArrayObject(String arrayString) {
    String[] floatArray = arrayString.split("\\[\\[")[1].split("]]");
    floatArray = floatArray[0].split(",");
    float[][] finalArr = new float[1][OUTPUT_SIZE];
    for (int i = 0, floatArrayLength = floatArray.length; i < floatArrayLength; i++) {
      String f = floatArray[i];
      float f1 = Float.parseFloat(f);
      finalArr[0][i] = f1;
    }
    return finalArr;
  }

  private void initializeBt(){
    manager = (BluetoothManager) getApplicationContext().getSystemService(
            Context.BLUETOOTH_SERVICE);
    btAdapter = manager.getAdapter();
    if (btAdapter == null) {
      Log.e("Bluetooth Error", "Bluetooth not detected on device");
    } else if (!btAdapter.isEnabled()) {
      Log.e("Error","Need to request Bluetooth");
      Intent enableBtIntent = new Intent(BluetoothAdapter.ACTION_REQUEST_ENABLE);
      this.startActivityForResult(enableBtIntent, REQUEST_ENABLE_BLUETOOTH);
    } else if (!btAdapter.isMultipleAdvertisementSupported()) {
      Log.e("Not supported", "BLE advertising not supported on this device");
    }
    bluetoothLeScanner = btAdapter.getBluetoothLeScanner();
    settings = new ScanSettings.Builder()
            .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
            .setLegacy(false)
            .setPhy(ScanSettings.PHY_LE_ALL_SUPPORTED)
            .setNumOfMatches(ScanSettings.MATCH_NUM_MAX_ADVERTISEMENT)
            .setCallbackType(ScanSettings.CALLBACK_TYPE_ALL_MATCHES)
            .setMatchMode(ScanSettings.MATCH_MODE_AGGRESSIVE)
            .build();
    filters = new ArrayList<ScanFilter>();
    byte[] test = new byte[200];
    byte[] mask = new byte [200];
    for (int i = 0; i < 200; i++){
      test[i] = (byte)1;
      mask[i] = (byte)0;
    }
    //filters.add(new ScanFilter.Builder().setServiceData(SERVICE_UUID,test,mask).build());
    filters.add(new ScanFilter.Builder().setServiceUuid(SERVICE_UUID).build());
  }

  private void showAddFaceDialog(SimilarityClassifier.Recognition rec) {

    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    LayoutInflater inflater = getLayoutInflater();
    View dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null);
    ImageView ivFace = dialogLayout.findViewById(R.id.dlg_image);
    TextView tvTitle = dialogLayout.findViewById(R.id.dlg_title);
    EditText etName = dialogLayout.findViewById(R.id.dlg_input);

    tvTitle.setText("Add Face");
    ivFace.setImageBitmap(rec.getCrop());
    etName.setHint("Input name");

    builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
      @Override
      public void onClick(DialogInterface dlg, int i) {

          String name = etName.getText().toString();
          if (name.isEmpty()) {
              return;
          }
          detector.register(name, rec);

          Intent intent = new Intent(DetectorActivity.this, BeaconBroadcast.class);
          //intent.putExtra("name", name);
//          BeaconBroadcast bb = new BeaconBroadcast();
//          Fragmenter.advertise(bb.adv,245,name.getBytes(),SERVICE_UUID,bb.parameters.build(),bb.callback);
          ByteArrayOutputStream bos = new ByteArrayOutputStream();
          ObjectOutputStream out = null;
          try {
            Log.v("DetectorActivity 588 Debug", "Debug");
            out = new ObjectOutputStream(bos);
            Log.v("DetectorActivity 590 Debug", "Debug");
            //Map.Entry<String, String> faceData = new AbstractMap.SimpleEntry(name, rec.toString());
            Log.v("here9", name.toString());
            Log.v("here10", rec.toString());
            //SimilarityClassifier.Recognition recs = new SimilarityClassifier.Recognition(rec.getId(),rec.getTitle(),rec.getDistance(), rec.getLocation());
            out.writeObject(name);
            Log.v("DetectorActivity 592 Debug", "Debug");
            out.flush();
            Log.v("DetectorActivity 594 Debug", "Debug");
            byte[] recBytes = bos.toByteArray();
            Log.v("DetectorActivity 596 Debug", "Debug");
            intent.putExtra("name", name.getBytes());
            intent.putExtra("id", rec.getId().getBytes());
            intent.putExtra("title",rec.getTitle().getBytes());
            intent.putExtra("dist",ByteBuffer.allocate(4).putFloat(rec.getDistance()).array());
            intent.putExtra("locL",ByteBuffer.allocate(4).putFloat(rec.getLocation().left).array());
            intent.putExtra("locT",ByteBuffer.allocate(4).putFloat(rec.getLocation().top).array());
            intent.putExtra("locR",ByteBuffer.allocate(4).putFloat(rec.getLocation().right).array());
            intent.putExtra("locB",ByteBuffer.allocate(4).putFloat(rec.getLocation().bottom).array());

            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            rec.getCrop().compress(Bitmap.CompressFormat.PNG, 100, stream);
            byte[] cropBA = stream.toByteArray();
            rec.getCrop().recycle();

            intent.putExtra("crop",cropBA);
            Log.v("here15", recBytes.toString());
            Log.v("DetectorActivity 598 Debug", "Debug");
          } catch (Exception ex) {
            Log.v("DetectorActivity 600 Debug", ex.getLocalizedMessage());
          } finally {
            try {
              bos.close();
            } catch (IOException ex) {
              Log.v("here","here16");
            }
          }
          startService(intent);

//          intent.putExtra("recId", rec.getId());
//          intent.putExtra("recTitle", rec.getTitle());
//          intent.putExtra("recDistance", rec.getDistance());
//          intent.putExtra("recLocation", rec.getLocation());
//          intent.putExtra("recColor", rec.getColor());
//          intent.putExtra("recExtra", (Bundle) rec.getExtra());
//          intent.putExtra("recCrop", rec.getCrop());

          Context context = getApplicationContext();
          CharSequence text = "BLE Broadcast: User Feature!";
          int duration = Toast.LENGTH_SHORT;

          Toast toast = Toast.makeText(context, text, duration);
          toast.show();

          //knownFaces.put(name, rec);
          dlg.dismiss();
      }
    });
    builder.setView(dialogLayout);
    builder.show();

  }

  private void updateResults(long currTimestamp, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

    tracker.trackResults(mappedRecognitions, currTimestamp);
    trackingOverlay.postInvalidate();
    computingDetection = false;
    //adding = false;

    if (mappedRecognitions.size() > 0) {
       LOGGER.i("Adding results");
       SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
       if (rec.getExtra() != null) {
         showAddFaceDialog(rec);
       }

    }

    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                showFrameInfo(previewWidth + "x" + previewHeight);
                showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                showInference(lastProcessingTimeMs + "ms");
              }
            });

  }

  private void onFacesDetected(long currTimestamp, List<Face> faces, boolean add) {

    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
    final Canvas canvas = new Canvas(cropCopyBitmap);
    final Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(2.0f);

    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
    switch (MODE) {
      case TF_OD_API:
        minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        break;
    }

    final List<SimilarityClassifier.Recognition> mappedRecognitions =
            new LinkedList<SimilarityClassifier.Recognition>();


    //final List<Classifier.Recognition> results = new ArrayList<>();

    // Note this can be done only once
    int sourceW = rgbFrameBitmap.getWidth();
    int sourceH = rgbFrameBitmap.getHeight();
    int targetW = portraitBmp.getWidth();
    int targetH = portraitBmp.getHeight();
    Matrix transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation);
    final Canvas cv = new Canvas(portraitBmp);

    // draws the original image in portrait mode.
    cv.drawBitmap(rgbFrameBitmap, transform, null);

    final Canvas cvFace = new Canvas(faceBmp);

    boolean saved = false;

    for (Face face : faces) {

      LOGGER.i("FACE" + face.toString());
      LOGGER.i("Running detection on face " + currTimestamp);
      //results = detector.recognizeImage(croppedBitmap);

      final RectF boundingBox = new RectF(face.getBoundingBox());

      //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
      final boolean goodConfidence = true; //face.get;
      if (boundingBox != null && goodConfidence) {

        // maps crop coordinates to original
        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
        transform.mapRect(faceBB);

        // translates portrait to origin and scales to fit input inference size
        //cv.drawRect(faceBB, paint);
        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
        Matrix matrix = new Matrix();
        matrix.postTranslate(-faceBB.left, -faceBB.top);
        matrix.postScale(sx, sy);

        cvFace.drawBitmap(portraitBmp, matrix, null);

        //canvas.drawRect(faceBB, paint);

        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

        if (add) {
          crop = Bitmap.createBitmap(portraitBmp,
                            (int) faceBB.left,
                            (int) faceBB.top,
                            (int) faceBB.width(),
                            (int) faceBB.height());
          //Yunqi
          Log.v("DetectorActivity 531", crop.toString());
        }

        final long startTime = SystemClock.uptimeMillis();
        final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, add);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

        if (resultsAux.size() > 0) {

          SimilarityClassifier.Recognition result = resultsAux.get(0);

          extra = result.getExtra();
//          Object extra = result.getExtra();
//          if (extra != null) {
//            LOGGER.i("embeeding retrieved " + extra.toString());
//          }

          float conf = result.getDistance();
          if (conf < 1.0f) {

            confidence = conf;
            label = result.getTitle();
            if (result.getId().equals("0")) {
              color = Color.GREEN;
            }
            else {
              color = Color.RED;
            }
          }

        }

        if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {

          // camera is frontal so the image is flipped horizontally
          // flips horizontally
          Matrix flip = new Matrix();
          if (sensorOrientation == 90 || sensorOrientation == 270) {
            flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          else {
            flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
          flip.mapRect(boundingBox);

        }

        final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                "0", label, confidence, boundingBox);

        result.setColor(color);
        result.setLocation(boundingBox);
        result.setExtra(extra);
        result.setCrop(crop);
        mappedRecognitions.add(result);

      }


    }

    //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }

    updateResults(currTimestamp, mappedRecognitions);


  }


}
