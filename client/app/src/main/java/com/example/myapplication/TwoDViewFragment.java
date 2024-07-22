package com.example.myapplication;

import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import com.mapbox.maps.MapView;
import com.mapbox.maps.Style;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link TwoDViewFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
public class TwoDViewFragment extends Fragment {

    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
    private static final String ARG_PARAM1 = "param1";
    private static final String ARG_PARAM2 = "param2";

    // TODO: Rename and change types of parameters
    private String mParam1;
    private String mParam2;

    MapView mapView;

    public TwoDViewFragment() {
        // Required empty public constructor
    }

    /**
     * Use this factory method to create a new instance of
     * this fragment using the provided parameters.
     *
     * @param param1 Parameter 1.
     * @param param2 Parameter 2.
     * @return A new instance of fragment TwoDViewFragment.
     */
    // TODO: Rename and change types and number of parameters
    public static TwoDViewFragment newInstance(String param1, String param2) {
        TwoDViewFragment fragment = new TwoDViewFragment();
        Bundle args = new Bundle();
        args.putString(ARG_PARAM1, param1);
        args.putString(ARG_PARAM2, param2);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            mParam1 = getArguments().getString(ARG_PARAM1);
            mParam2 = getArguments().getString(ARG_PARAM2);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_two_d_view, container, false);
        mapView = view.findViewById(R.id.mapView);
        mapView.getMapboxMap().loadStyle("mapbox://styles/wjc1512/clxhrmx8m008e01qw5roffj8p");
        return view;
    }
}