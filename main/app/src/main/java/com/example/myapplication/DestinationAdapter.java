package com.example.myapplication;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;


import java.util.List;

public class DestinationAdapter extends RecyclerView.Adapter<DestinationAdapter.ViewHolder> {

    private List<String> destinations;
    private OnItemClickListener listener;

    public interface OnItemClickListener {
        void onItemClick(String destination);
    }

    public DestinationAdapter(List<String> destinations, OnItemClickListener listener) {
        this.destinations = destinations;
        this.listener = listener;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_destination, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        String destination = destinations.get(position);
        holder.destinationName.setText(destination);
        holder.itemView.setOnClickListener(v -> listener.onItemClick(destination));
    }

    @Override
    public int getItemCount() {
        return destinations.size();
    }

    public void filterList(List<String> filteredList) {
        destinations = filteredList;
        notifyDataSetChanged();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView destinationName;

        ViewHolder(View itemView) {
            super(itemView);
            destinationName = itemView.findViewById(R.id.destinationName);
        }
    }
}
