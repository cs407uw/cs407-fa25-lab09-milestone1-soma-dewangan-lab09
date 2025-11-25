package com.cs407.lab09

import android.hardware.Sensor
import android.hardware.SensorEvent
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class BallViewModel : ViewModel() {

    private var ball: Ball? = null
    private var lastTimestamp: Long = 0L

    // Expose the ball's position as a StateFlow
    private val _ballPosition = MutableStateFlow(Offset.Zero)
    val ballPosition: StateFlow<Offset> = _ballPosition.asStateFlow()

    /**
     * Called by the UI when the game field's size is known.
     */
    fun initBall(fieldWidth: Float, fieldHeight: Float, ballSizePx: Float) {
        if (ball == null) {
            // DONE: Initialize the ball instance
            // ball = Ball(...)
            ball = Ball(fieldWidth, fieldHeight, ballSizePx)

            // DONE: Update the StateFlow with the initial position
            // _ballPosition.value = Offset(ball!!.posX, ball!!.posY)
            _ballPosition.value = Offset(ball!!.posX, ball!!.posY)
        }
    }

    /**
     * Called by the SensorEventListener in the UI.
     */
    fun onSensorDataChanged(event: SensorEvent) {
        // Ensure ball is initialized
        val currentBall = ball ?: return

        if (event.sensor.type == Sensor.TYPE_GRAVITY) {
            if (lastTimestamp != 0L) {
                // DONE: Calculate the time difference (dT) in seconds
                // Hint: event.timestamp is in nanoseconds
                // val NS2S = 1.0f / 1000000000.0f
                // val dT = ...
                val NS2S = 1.0f / 1000000000.0f
                val dT = (event.timestamp - lastTimestamp) * NS2S

                // DONE: Update the ball's position and velocity
                // Hint: The sensor's x and y-axis are inverted
                // currentBall.updatePositionAndVelocity(xAcc = ..., yAcc = ..., dT = ...)
                currentBall.updatePositionAndVelocity(
                    xAcc = -event.values[0],
                    yAcc = event.values[1],
                    dT = dT
                )

                // DONE: Update the StateFlow to notify the UI
                // _ballPosition.update { Offset(currentBall.posX, currentBall.posY) }
                _ballPosition.update { Offset(currentBall.posX, currentBall.posY) }
            }

            // DONE: Update the lastTimestamp
            // lastTimestamp = ...
            lastTimestamp = event.timestamp
        }
    }

    fun reset() {
        // DONE: Reset the ball's state
        // ball?.reset()
        ball?.reset()

        // DONE: Update the StateFlow with the reset position
        // ball?.let { ... }
        ball?.let {
            _ballPosition.value = Offset(it.posX, it.posY)
        }

        // DONE: Reset the lastTimestamp
        // lastTimestamp = 0L
        lastTimestamp = 0L
    }
}