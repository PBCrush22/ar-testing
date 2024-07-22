package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import androidx.appcompat.widget.SearchView;

import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentContainerView;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.navigation.NavigationView;
import com.google.android.material.snackbar.Snackbar;


import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private FragmentContainerView fragmentContainerView;
    private Fragment cameraFragment;
    private Fragment twoDViewFragment;
    private Fragment activeFragment;
    // Define the permission request code
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1001;

    private RecyclerView recyclerView;
    private DestinationAdapter adapter;
    private List<String> allDestinations;
    private List<String> filteredDestinations;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.drawer_layout);

        // Check if the camera permission is already granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            // You can use the camera here
            openCamera();
        } else {
            // Request the camera permission
            requestCameraPermission();
        }

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        DrawerLayout drawer = findViewById(R.id.drawer);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(
                this, drawer, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close
        );
        drawer.addDrawerListener(toggle);
        toggle.getDrawerArrowDrawable().setColor(getResources().getColor(R.color.white));
        toggle.syncState();



        fragmentContainerView = findViewById(R.id.frag_container);

        cameraFragment = new CameraViewFragment();
        twoDViewFragment = new TwoDViewFragment();

        // Set default fragment
        activeFragment = cameraFragment;
        getSupportFragmentManager().beginTransaction().replace(R.id.frag_container, activeFragment).commit();

        NavigationView navigationView = findViewById(R.id.navview);
        navigationView.setNavigationItemSelectedListener(item -> {
            int id = item.getItemId();

            if (id==R.id.voice_nav) {
                handleVoiceNavClick(item);
                // Perform action for turning off voice navigation
            }else if (id==R.id.twod_view){
                    // Perform action for opening 2D map
                handle2dViewClick(item);
            }

            // Close the drawer after handling item click
            drawer.closeDrawers();
            return true;
        });

        recyclerView = findViewById(R.id.recyclerView);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // Example list of destinations
        allDestinations = new ArrayList<>();
        allDestinations.add("New York");
        allDestinations.add("Los Angeles");
        allDestinations.add("Chicago");
        allDestinations.add("Houston");
        allDestinations.add("Phoenix");

        filteredDestinations = new ArrayList<>(allDestinations);
        adapter = new DestinationAdapter(filteredDestinations, this::onDestinationSelected);
        recyclerView.setAdapter(adapter);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });


    }

    // Method to request camera permission
    private void requestCameraPermission() {
        ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST_CODE);
    }

    // Handle the result of the permission request
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted
                openCamera();
            } else {
                // Permission denied
                Toast.makeText(this, "Camera permission is required to use the camera", Toast.LENGTH_SHORT).show();
            }
        }
    }

    // Method to open the camera (example method)
    private void openCamera() {
        // Implement your logic to open the camera here
        Toast.makeText(this, "Camera permission granted. Now you can use the camera.", Toast.LENGTH_SHORT).show();
    }

    private void handleVoiceNavClick(MenuItem item) {
        if (item.getTitle().equals(getString(R.string.menu_turn_off_voice))) {
            // Perform action for turning off voice navigation
            Toast.makeText(MainActivity.this, "Turning off voice navigation", Toast.LENGTH_SHORT).show();
            item.setTitle(getString(R.string.menu_turn_on_voice));
        } else {
            // Perform action for turning on voice navigation
            Toast.makeText(MainActivity.this, "Turning on voice navigation", Toast.LENGTH_SHORT).show();
            item.setTitle(getString(R.string.menu_turn_off_voice));
        }
    }

    private void handle2dViewClick(MenuItem item){
        if (item.getTitle().equals(getString(R.string.menu_open_2d_map))) {
            // Switch to 2D view
            switchFragment(twoDViewFragment);
            Toast.makeText(MainActivity.this, "Opening 2D map", Toast.LENGTH_SHORT).show();
            item.setTitle(getString(R.string.menu_close_2d_map));
//            inflate TwoDViewFragment
        } else {
            // Switch to camera view
            switchFragment(cameraFragment);
            Toast.makeText(MainActivity.this, "Closing 2D map", Toast.LENGTH_SHORT).show();
            item.setTitle(getString(R.string.menu_open_2d_map));
//            inflate CameraViewFragment

        }
    }

    private void switchFragment(Fragment fragment) {
        if (fragment != activeFragment) {
            FragmentManager fragmentManager = getSupportFragmentManager();
            FragmentTransaction fragmentTransaction = fragmentManager.beginTransaction();
            fragmentTransaction.replace(R.id.frag_container, fragment);
            fragmentTransaction.commit();
            activeFragment = fragment;
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.search_menu, menu);

        MenuItem searchItem = menu.findItem(R.id.action_search);
        SearchView searchView = (SearchView) searchItem.getActionView();

        // Set query hint text color
        EditText searchEditText = searchView.findViewById(androidx.appcompat.R.id.search_src_text);
        searchEditText.setHintTextColor(ContextCompat.getColor(this, R.color.typehint));

        // Remove default search icon
        ImageView searchIcon = searchView.findViewById(androidx.appcompat.R.id.search_button);
//        searchIcon.setColorFilter(R.color.colorPrimary);
        searchIcon.setVisibility(View.GONE); // or set visibility to INVISIBLE if you want to keep the space

//        back icon
        // Set navigation icon (back arrow) color
//        Toolbar toolbar = findViewById(R.id.toolbar);
//        toolbar.


        // Set close button color (if using a custom close icon drawable)
        ImageView closeButton = searchView.findViewById(androidx.appcompat.R.id.search_close_btn);
        Drawable closeDrawable = closeButton.getDrawable();
        if (closeDrawable != null) {
            closeDrawable.setColorFilter(ContextCompat.getColor(this, R.color.white), PorterDuff.Mode.SRC_ATOP);
        }

        // Set query text color
        searchEditText.setTextColor(ContextCompat.getColor(this, R.color.white));

        searchView.setQueryHint("Enter desired destination");

        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
//                if (predefinedLocations.contains(query)) {
//                    Toast.makeText(MainActivity.this, "Location found: " + query, Toast.LENGTH_SHORT).show();
//                } else {
//                    Toast.makeText(MainActivity.this, "Location not recognized", Toast.LENGTH_SHORT).show();
//                }
                return false;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                filter(newText);
                return false;
            }
        });

        searchItem.setOnActionExpandListener(new MenuItem.OnActionExpandListener() {
            @Override
            public boolean onMenuItemActionExpand(MenuItem item) {
                recyclerView.setVisibility(View.VISIBLE);
                return true;
            }

            @Override
            public boolean onMenuItemActionCollapse(MenuItem item) {
                recyclerView.setVisibility(View.GONE);
                return true;
            }
        });

        return true;
    }

    private void filter(String text) {
        filteredDestinations.clear();
        if (text.isEmpty()) {
            filteredDestinations.addAll(allDestinations);
        } else {
            for (String destination : allDestinations) {
                if (destination.toLowerCase().contains(text.toLowerCase())) {
                    filteredDestinations.add(destination);
                }
            }
        }
        adapter.filterList(filteredDestinations);
    }

    private void onDestinationSelected(String destination) {
        Toast.makeText(this, "Selected: " + destination, Toast.LENGTH_SHORT).show();
        recyclerView.setVisibility(View.GONE); // Hide the RecyclerView after selection

        Toolbar toolbar = findViewById(R.id.toolbar);
        // Collapse the search view and hide the keyboard
        MenuItem searchItem = toolbar.getMenu().findItem(R.id.action_search);
        if (searchItem.isActionViewExpanded()) {
            searchItem.collapseActionView();
            InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
            if (imm != null) {
                imm.hideSoftInputFromWindow(searchItem.getActionView().getWindowToken(), 0);
            }
        }

        View view = findViewById(R.id.main);
// Navigate back to CameraViewFragment and show AR arrows here
        Snackbar snackbar = Snackbar.make(view, destination, Snackbar.LENGTH_INDEFINITE);
        snackbar.setAction("STOP", v -> {
                    // Handle action button click here
                    snackbar.dismiss(); // Dismiss the Snackbar when action is clicked
                });

        View snackBarView = snackbar.getView();
        Button button=
                (Button) snackBarView.findViewById(com.google.android.material.R.id.snackbar_action);
        button.setBackgroundColor(getResources().getColor(R.color.colorPrimaryLight));

        // Get the color from resources
        int whiteColor = ContextCompat.getColor(this, R.color.white);

        // Create a ColorStateList with the specified color
        ColorStateList whiteTextColor = ColorStateList.valueOf(whiteColor);

        // Get the color from resources
        int green = ContextCompat.getColor(this, R.color.colorPrimary);

        // Create a ColorStateList with the specified color
        ColorStateList greenColor = ColorStateList.valueOf(green);

        snackbar.setTextColor(whiteTextColor);
        snackbar.setActionTextColor(whiteTextColor);
        snackbar.setBackgroundTint(green);

        snackbar.show(); // Show the Snackbar


    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.action_search) {
            // Handle search action
            return true;
        }

        return super.onOptionsItemSelected(item);
    }


}