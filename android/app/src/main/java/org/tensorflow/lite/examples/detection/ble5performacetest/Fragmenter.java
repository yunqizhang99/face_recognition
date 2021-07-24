package org.tensorflow.lite.examples.detection.ble5performacetest;

import android.bluetooth.le.AdvertiseData;
import android.bluetooth.le.AdvertisingSetCallback;
import android.bluetooth.le.AdvertisingSetParameters;
import android.bluetooth.le.BluetoothLeAdvertiser;
import android.os.Build;
import android.os.ParcelUuid;
import android.util.Log;

import androidx.annotation.RequiresApi;

import java.util.Arrays;

public class Fragmenter {
    private static final String TAG = "FRAGMENTER";
    private static final int ADV_TIME = 420;
    private static final int ADV_TIME_SINGLE = 10000;
    private static boolean advertise_flag = true;
    private static long endTime;


    @RequiresApi(api = Build.VERSION_CODES.O)
    public static void advertise(final BluetoothLeAdvertiser adv, int max_size, byte[] data, ParcelUuid uuid, final AdvertisingSetParameters advertiseSettings, final AdvertisingSetCallback advertiseCallback){
        int full_packet_count = data.length / (max_size-2);
        int last_packet_bytes = data.length % (max_size-2);
        int packet_num = 1;
        byte[] adv_packet;
        int loopcount = 0;
        long minTime = System.currentTimeMillis() + 20000;

        while ((advertise_flag) && (loopcount < 5  || System.currentTimeMillis() < minTime) ){
            adv_packet = new byte[max_size];
            if (full_packet_count == 0){
                //adv_packet = new byte[data.length+2];
                adv_packet[0] = (byte)1;
                adv_packet[1] = (byte)1;
                for (int i = 0; i < data.length; i++){
                    adv_packet[i+2] = data[i];
                }
                AdvertiseData advertiseData = new AdvertiseData.Builder()
                        .addServiceData(uuid,adv_packet)
                        .addServiceUuid(uuid)
                        .setIncludeTxPowerLevel(false)
                        .setIncludeDeviceName(true)
                        .build();
                adv.startAdvertisingSet(advertiseSettings, advertiseData, null, null, null, advertiseCallback);
                endTime = System.currentTimeMillis() + ADV_TIME_SINGLE;
                while (System.currentTimeMillis() < endTime){
                }
                adv.stopAdvertisingSet(advertiseCallback);
            }
            else {
                while (packet_num <= full_packet_count){
                    if (packet_num == 1){
                        adv_packet[0] = (byte)1;
                    } else {
                        adv_packet[0] = (byte)packet_num;
                    }
                    if (packet_num == full_packet_count && last_packet_bytes == 0){
                        adv_packet[1] = (byte)1;
                    } else {
                        adv_packet[1] = (byte)0;
                    }
                    byte[] sub = Arrays.copyOfRange(data,(max_size - 2)*(packet_num - 1),(max_size - 2)*packet_num);
                    for (int j = 0; j < sub.length; j++){
                        adv_packet[j+2] = sub[j];
                    }
                    AdvertiseData advertiseData = new AdvertiseData.Builder()
                            .addServiceData(uuid,adv_packet)
                            .addServiceUuid(uuid)
                            .setIncludeTxPowerLevel(false)
                            .setIncludeDeviceName(true)
                            .build();

                    adv.startAdvertisingSet(advertiseSettings, advertiseData, null, null, null, advertiseCallback);
                    Log.i("Packet Number: ", Integer.toString(packet_num));
                    endTime = System.currentTimeMillis() + ADV_TIME;
                    while (System.currentTimeMillis() < endTime){
                    }
                    adv.stopAdvertisingSet(advertiseCallback);
                    packet_num++;
                }
                if (last_packet_bytes != 0){
                    //adv_packet = new byte[last_packet_bytes+2];
                    adv_packet[0] = (byte)packet_num;
                    adv_packet[1] = (byte)1;
                    for (int k = 0; k < last_packet_bytes; k++){
                        adv_packet[k+2] = data[full_packet_count*(max_size - 2) + k];
                    }
                    for (int m = last_packet_bytes+2; m < max_size; m++){
                        adv_packet[m] = (byte)0;
                    }
                    AdvertiseData advertiseData = new AdvertiseData.Builder()
                            .addServiceData(uuid,adv_packet)
                            .addServiceUuid(uuid)
                            .setIncludeTxPowerLevel(false)
                            .setIncludeDeviceName(true)
                            .build();

                    adv.startAdvertisingSet(advertiseSettings, advertiseData, null, null, null, advertiseCallback);
                    endTime = System.currentTimeMillis() + ADV_TIME;
                    Log.i("Packet Number: ","Last Packet");
                    while (System.currentTimeMillis() < endTime){
                    }
                    adv.stopAdvertisingSet(advertiseCallback);
                }
                packet_num = 1;
            }
            loopcount++;
        }
        adv.stopAdvertisingSet(advertiseCallback);

    }

    public static void setAdvertiseFlag(boolean set){
        advertise_flag = set;
    }
}
