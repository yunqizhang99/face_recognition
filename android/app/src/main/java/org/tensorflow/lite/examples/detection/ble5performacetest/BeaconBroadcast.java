package org.tensorflow.lite.examples.detection.ble5performacetest;

import android.app.IntentService;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.le.AdvertiseCallback;
import android.bluetooth.le.AdvertiseSettings;
import android.bluetooth.le.AdvertisingSet;
import android.bluetooth.le.AdvertisingSetCallback;
import android.bluetooth.le.AdvertisingSetParameters;
import android.bluetooth.le.BluetoothLeAdvertiser;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.ParcelUuid;
import android.util.Log;

import androidx.annotation.RequiresApi;

import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;

import java.io.Serializable;
import java.nio.Buffer;
import java.nio.ByteBuffer;

public class BeaconBroadcast extends IntentService {

    protected static final String TAG = "BeaconBroadcast";
    private static final ParcelUuid SERVICE_UUID =
            ParcelUuid.fromString("0000FEAA-0000-1000-8000-00805F9B34FB");

    private BluetoothManager manager;
    private BluetoothAdapter btAdapter;
    private BluetoothLeAdvertiser adv;
    private AdvertiseSettings advertiseSettings;
    private AdvertiseCallback advertiseCallback;

    AdvertisingSetParameters.Builder parameters = (new AdvertisingSetParameters.Builder())
            .setLegacyMode(false)
            .setInterval(AdvertisingSetParameters.INTERVAL_HIGH)
            .setTxPowerLevel(AdvertisingSetParameters.TX_POWER_MEDIUM)
            .setPrimaryPhy(BluetoothDevice.PHY_LE_1M)
            .setSecondaryPhy(BluetoothDevice.PHY_LE_2M);

    AdvertisingSetCallback callback = new AdvertisingSetCallback() {
        @Override
        public void onAdvertisingSetStarted(AdvertisingSet advertisingSet, int txPower, int status) {
            Log.i("BLE 5", "onAdvertisingSetStarted(): txPower:" + txPower + " , status: "
                    + status);
        }

        @Override
        public void onAdvertisingSetStopped(AdvertisingSet advertisingSet) {
            Log.i("BLE 5", "onAdvertisingSetStopped():");
        }
    };

    public BeaconBroadcast() {
        super(TAG);
        Log.v("here","here17");
    }

    @Override
    public void onCreate() {
        super.onCreate();
        initializeBT();
    }

    @Override
    protected void onHandleIntent(Intent intent) {
        if (intent != null) {
            Log.v("here","here12");
            byte [] name = intent.getByteArrayExtra("name");
            byte [] id = intent.getByteArrayExtra("id");
            byte [] title = intent.getByteArrayExtra("title");
            byte [] dist = intent.getByteArrayExtra("dist");
            byte [] locL = intent.getByteArrayExtra("locL");
            byte [] locT = intent.getByteArrayExtra("locT");
            byte [] locR = intent.getByteArrayExtra("locR");
            byte [] locB = intent.getByteArrayExtra("locB");
            byte [] crop = intent.getByteArrayExtra("crop");

            String nameS = new String(name);
            Log.v("here", "here24 "+nameS);

            byte [] separator = "*".getBytes();

            byte[] finalBA = new byte[name.length + separator.length + id.length + separator.length + title.length + separator.length + dist.length + separator.length + locL.length + separator.length + locT.length + separator.length + locR.length + separator.length + locB.length + separator.length + crop.length];
            System.arraycopy(name, 0, finalBA, 0, name.length);
            System.arraycopy(separator, 0, finalBA, name.length, separator.length);
            System.arraycopy(id, 0, finalBA, name.length+separator.length, id.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length, separator.length);
            System.arraycopy(title, 0, finalBA, name.length+separator.length+id.length+separator.length, title.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length, separator.length);
            System.arraycopy(dist, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length, dist.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length, separator.length);
            System.arraycopy(locL, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length, locL.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length, separator.length);
            System.arraycopy(locT, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length, locT.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length+locT.length, separator.length);
            System.arraycopy(locR, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length+locT.length+separator.length, locR.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length+locT.length+separator.length+locR.length, separator.length);
            System.arraycopy(locB, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length+locT.length+separator.length+locR.length+separator.length, locB.length);
            System.arraycopy(separator, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length+locT.length+separator.length+locR.length+separator.length+locB.length, separator.length);
            System.arraycopy(crop, 0, finalBA, name.length+separator.length+id.length+separator.length+title.length+separator.length+dist.length+separator.length+locL.length+separator.length+locT.length+separator.length+locR.length+separator.length+locB.length+separator.length, crop.length);

            Log.v("here", "here18");
            Log.v("here", "here20 "+finalBA.length);
            Fragmenter.advertise(adv,245,finalBA,SERVICE_UUID,parameters.build(),callback);
            Log.v("here", "here19");
        }
    }

    private void initializeBT(){
        manager = (BluetoothManager) getApplicationContext().getSystemService(
                Context.BLUETOOTH_SERVICE);
        btAdapter = manager.getAdapter();
        if (btAdapter == null) {
            Log.e("Bluetooth Error", "Bluetooth not detected on device");
        } else if (!btAdapter.isEnabled()) {
            Log.e("Error","Need Bluetooth Permissions");
        } else if (!btAdapter.isMultipleAdvertisementSupported()) {
            Log.e("Not supported", "BLE advertising not supported on this device");
        }
        adv = btAdapter.getBluetoothLeAdvertiser();
        if (!btAdapter.isLe2MPhySupported()) {
            Log.e("BLE 5", "2M PHY not supported!");
            return;
        }
        if (!btAdapter.isLeExtendedAdvertisingSupported()) {
            Log.e("BLE 5", "LE Extended Advertising not supported!");
            return;
        }
        advertiseSettings = new AdvertiseSettings.Builder()
                .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
                .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_MEDIUM)
                .setConnectable(true)
                .build();
        advertiseCallback = createAdvertiseCallback();
    }

    private AdvertiseCallback createAdvertiseCallback() {
        return new AdvertiseCallback() {

            @Override

            public void onStartFailure(int errorCode) {
                switch (errorCode) {
                    case ADVERTISE_FAILED_DATA_TOO_LARGE:
                        Log.e("Failure","ADVERTISE_FAILED_DATA_TOO_LARGE");
                        break;
                    case ADVERTISE_FAILED_TOO_MANY_ADVERTISERS:
                        Log.e("Failure","ADVERTISE_FAILED_TOO_MANY_ADVERTISERS");
                        break;
                    case ADVERTISE_FAILED_ALREADY_STARTED:
                        Log.e("Failure","ADVERTISE_FAILED_ALREADY_STARTED");
                        break;
                    case ADVERTISE_FAILED_INTERNAL_ERROR:
                        Log.e("Failure","ADVERTISE_FAILED_INTERNAL_ERROR");
                        break;
                    case ADVERTISE_FAILED_FEATURE_UNSUPPORTED:
                        Log.e("Failure","ADVERTISE_FAILED_FEATURE_UNSUPPORTED");
                        break;
                    default:
                        Log.e("Failure","startAdvertising failed with unknown error " + errorCode);
                        break;
                }
            }

        };
    }

}
