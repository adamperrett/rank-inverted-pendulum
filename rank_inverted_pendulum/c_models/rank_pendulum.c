// Standard includes
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>
#include "random.h"
#include <stdfix.h>
#include <math.h>
#include <common/maths-util.h>

#include <recording.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------

// Frame delay (ms)
//#define time_increment 200 //14//20
/*
    number of bins for current angle of the pole
    number of bins for the force to be applied or number of spikes per tick equals a force
    mass of the cart
    mass of the pole
    initial starting angle
    velocity of the cart
    velocity of the pendulum
    base rate for the neurons to fire in each bin
    each spike equals a change in force to be applied (what is that amount)
    receptive field of each bin
    update model on each timer tick and on each spike received, or number of spikes per tick equals a force

    add option to rate (increased poisson P()) code and rank code
*/

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_PENDULUM,
  REGION_RECORDING,
  REGION_DATA,
} region_t;

typedef enum
{
  SPECIAL_EVENT_ANGLE,
  SPECIAL_EVENT_ANGLE_V,
  SPECIAL_EVENT_CART,
  SPECIAL_EVENT_CART_V,
} special_event_t;

typedef enum // forward will be considered positive motion
{
  BACKWARD_MOTOR  = 0x0,
  FORWARD_MOTOR  = 0x1,
} arm_key_t;

typedef union{
   uint32_t u;
//   uint32_t* us;
   float f;
   accum a;
} uint_float_union;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

mars_kiss64_seed_t kiss_seed;

int32_t current_score = 0;
int32_t reward_based = 1;

// experimental constraints and variables
float current_time = 0;
float max_motor_force = 10; // N
float min_motor_force = -10; // N
float motor_force = 0;
float force_increment = 100;
bool read_force = true;
float track_length = 4.8; // m
float cart_position = 0; // m
float cart_velocity = 0;  // m/s
float cart_acceleration = 0;  // m/s^2
float highend_cart_v = 5; // used to calculate firing rate and bins
float max_pole_angle = (36.0f / 180.f) * M_PI;
float min_pole_angle = -(36.0f / 180.f) * M_PI;
float max_pole_angle_bin = (36.0f / 180.f) * M_PI;
float min_pole_angle_bin = -(36.0f / 180.f) * M_PI;
uint_float_union pole_angle_accum;
float pole_angle;
float pole_velocity = 0; // angular/s
float pole_acceleration = 0; // angular/s^2
float highend_pole_v = 5; // used to calculate firing rate and bins

#define max_bins 10
float pole_angle_spike_time[max_bins] = {0.f};
float pole_velocity_spike_time[max_bins] = {0.f};
float cart_position_spike_time[max_bins] = {0.f};
float cart_velocity_spike_time[max_bins] = {0.f};

int max_firing_rate = 20;
float max_firing_prob = 0;
int encoding_scheme; // 0 = upcapped time, 1 = capped time
int number_of_bins = 20;
float bin_width;
float bin_overlap = 2.5;
float tau_force;
uint_float_union temp_accum;

int central = 1; // if it's central that mean perfectly central on the track and angle is the lowest rate, else half

// experimental parameters
uint_float_union half_pole_length_accum; // m
float half_pole_length; // m

float max_balance_time = 0;

float current_state[2];
bool in_bounds = true;

uint32_t time_increment;

//! How many ticks until next frame
static uint32_t tick_in_frame = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;
uint32_t score_change_count=0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void spike_angle(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_ANGLE * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_angle \t%d - \t%u\n", bin, mask);
}

static inline void spike_angle_v(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_ANGLE_V * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_angle_v \t%d - \t%u\n", bin, mask);
}

static inline void spike_cart(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_CART * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_cart \t%d - \t%u\n", bin, mask);
}

static inline void spike_cart_v(int bin)
{
    uint32_t mask;
    mask = (SPECIAL_EVENT_CART_V * number_of_bins) + bin;
    spin1_send_mc_packet(key | (mask), 0, NO_PAYLOAD);
//    io_printf(IO_BUF, "spike_cart_v \t%d - \t%u\n", bin, mask);
}

void resume_callback() {
    recording_reset();
}

//void add_event(int i, int j, colour_t col, bool bricked)
//{
//  const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
//  const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << 10) + (j << 2) + (bricked<<1) + colour_bit);
//
//  spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "%d, %d, %u, %08x\n", i, j, col, spike_key);
//}

static bool initialize(uint32_t *timer_period)
{
//    io_printf(IO_BUF, "Initialise bandit: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address))
    {
      return false;
    }
    /*
    simulation_initialise(
        address_t address, uint32_t expected_app_magic_number,
        uint32_t* timer_period, uint32_t *simulation_ticks_pointer,
        uint32_t *infinite_run_pointer, int sdp_packet_callback_priority,
        int dma_transfer_done_callback_priority)
    */
    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL))
    {
      return false;
    }
//    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read breakout region
    address_t breakout_region = data_specification_get_region(REGION_PENDULUM, address);
    key = breakout_region[0];
//    io_printf(IO_BUF, "\tKey=%08x\n", key);
//    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    //get recording region
    address_t recording_address = data_specification_get_region(
                                       REGION_RECORDING,address);
    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(recording_address, &recording_flags))
    {
       rt_error(RTE_SWERR);
       return false;
    }

    cart_position = track_length / 2;

    address_t pend_region = data_specification_get_region(REGION_DATA, address);
//    encoding_scheme = pend_region[0]; // 0 rate
    encoding_scheme = pend_region[0];
    time_increment = pend_region[1];
    half_pole_length_accum.u = pend_region[2];
//    half_pole_length = (float)(half_pole_length_accum.a);
//    io_printf(IO_BUF, "half %u, norm %k, half %f\n", (accum)half_pole_length, (accum)half_pole_length, (accum)half_pole_length);
//    half_pole_length_accum.a = half_pole_length_accum.a / 2.0k;
//    half_pole_length = half_pole_length_accum.f / 2.0f;
//    io_printf(IO_BUF, "half %u, norm %k, half %f\n", (accum)half_pole_length, (accum)half_pole_length, (accum)half_pole_length);
//    half_pole_length = half_pole_length_accum.u / 2.0f;
//    io_printf(IO_BUF, "half %u, norm %k, half %f\n", (accum)half_pole_length, (accum)half_pole_length, (accum)half_pole_length);
    half_pole_length = half_pole_length_accum.a / 2.0f;
//    io_printf(IO_BUF, "half %u, norm %k, half %f\n", (accum)half_pole_length, (accum)half_pole_length, (accum)half_pole_length);
    pole_angle_accum.u = pend_region[3];
    pole_angle = pole_angle_accum.a;
//    io_printf(IO_BUF, "angle d %k\n", (accum)pole_angle);
//    io_printf(IO_BUF, "pi %k\n", (accum)M_PI);
//    io_printf(IO_BUF, "180 %k\n", (accum)(pole_angle / 180.0f));
    pole_angle = (pole_angle / 180.0f) * M_PI;
//    io_printf(IO_BUF, "angle r %k\n", (accum)pole_angle);
//    pole_angle = (float)(pole_angle_accum.a);
//    accum temp_angle = pend_region[3];
//    float new_angle = (float)(temp_angle);
//    accum newer_angle = (accum)(pend_region[3]);
//    accum test1 = 0.1k;
//    io_printf(IO_BUF, "angle %k, divided %k test %k\n", temp_angle, newer_angle, test1);
//    io_printf(IO_BUF, "good angle u %u, a %k, f %f\n", pole_angle_accum.u, pole_angle_accum.a, pole_angle_accum.f);
//    io_printf(IO_BUF, "angle u %u, a %k, f %f\n", pole_angle, pole_angle, pole_angle);
//    pole_angle = (float)pend_region[3]; // ((float)pend_region[3] / (float)0xffffffff); //
    reward_based = pend_region[4];
    force_increment = pend_region[5]; // (float)pend_region[5] / (float)0xffff;
    max_firing_rate = pend_region[6];
    number_of_bins = pend_region[7];
    central = pend_region[8];

    bin_width = 1.f / ((float)number_of_bins - 1.f);
    max_firing_prob = max_firing_rate / 1000.f;
//    accum
    // pass in random seeds
    kiss_seed[0] = pend_region[9];
    kiss_seed[1] = pend_region[10];
    kiss_seed[2] = pend_region[11];
    kiss_seed[3] = pend_region[12];
    validate_mars_kiss64_seed(kiss_seed);

    temp_accum.u = pend_region[13];
    bin_overlap = temp_accum.a;
    temp_accum.u = pend_region[14];
    tau_force = temp_accum.a;

    force_increment = (float)((max_motor_force - min_motor_force) / (float)force_increment);

//    io_printf(IO_BUF, "starting state (d,v,a):(%k, %k, %k) and cart (d,v,a):(%k, %k, %k)\n", (accum)pole_angle, (accum)pole_velocity,
//                        (accum)pole_acceleration, (accum)cart_position, (accum)cart_velocity, (accum)cart_acceleration);

//    io_printf(IO_BUF, "Initialise: completed successfully\n");

//    auto start = chrono::steady_clock::now();
    return true;
}

float firing_time(float relative_value, int bin){
    float separation = relative_value - (bin_width * (float)bin);
    separation = separation * bin_overlap;
    float maximum_time_window = 1000.f / (float)max_firing_rate;
    float delay;
    if (separation < 0.f){
        separation = separation * -1.f;
    }
//    if (separation < 1.f){
//        delay = maximum_time_window * separation;
//    }
    delay = maximum_time_window * separation;
    if (delay > time_increment){
        if (encoding_scheme == 1){
            delay = time_increment;
        }
    }
//    io_printf(IO_BUF, "(r, se, b, d, m):(%k, %k, %k, %k, %k)\n", (accum)relative_value, (accum)separation,
//                        (accum)bin, (accum)delay, (accum)maximum_time_window);
    return delay;
}

// updates the current state of the pendulum
bool update_state(float time_step){
    float gravity = -9.8; // m/s^2
    float mass_cart = 1; // kg
    float mass_pole = 0.1; // kg
    float friction_cart_on_track = 0.0005; // coefficient of friction
    float friction_pole_hinge = 0.000002; // coefficient of friction

    float effective_force_pole_on_cart = 0;
    float pole_angle_force = (mass_pole * half_pole_length * pole_velocity * pole_velocity * sin(pole_angle));
    float angle_scalar = ((3.0f / 4.0f) * mass_pole * cos(pole_angle));
    float friction_and_gravity = (((friction_pole_hinge * pole_velocity) / (mass_pole * half_pole_length)) +
                        (gravity * sin(pole_angle)));
    float effective_pole_mass = mass_pole * (1.0f - ((3.0f / 4.0f) * cos(pole_angle) * cos(pole_angle)));

    effective_force_pole_on_cart = pole_angle_force + (angle_scalar * friction_and_gravity);
    if (cart_velocity > 0){
        cart_acceleration = (motor_force - friction_cart_on_track + effective_force_pole_on_cart) /
                                (mass_cart + effective_pole_mass);
    }
    else{
        cart_acceleration = (motor_force + friction_cart_on_track + effective_force_pole_on_cart) /
                                (mass_cart + effective_pole_mass);
    }

    float length_scalar = -3.0f / (4.0f * half_pole_length);
    float cart_acceleration_effect = cart_acceleration * cos(pole_angle);
    float gravity_effect = gravity * sin(pole_angle);
    float friction_effect = (friction_pole_hinge * pole_velocity) / (mass_pole * half_pole_length);
    pole_acceleration = length_scalar * (cart_acceleration_effect + gravity_effect + friction_effect);

    cart_velocity = (cart_acceleration * time_step) + cart_velocity;
    cart_position = (cart_velocity * time_step) + cart_position;

    pole_velocity = (pole_acceleration * time_step) + pole_velocity;
    pole_angle = (pole_velocity * time_step) + pole_angle;

    float relative_cart;
    float relative_angle;
    float relative_cart_velocity;
    float relative_angular_velocity;
    relative_angle = (pole_angle + max_pole_angle_bin) / (2.f * max_pole_angle_bin);
    relative_angular_velocity = (pole_velocity + highend_pole_v) / (2.f * highend_pole_v);
    relative_cart = cart_position / track_length;
    relative_cart_velocity = (cart_velocity + highend_cart_v) / (2.f * highend_cart_v);
    for (int i = 0; i < number_of_bins; i = i + 1){
        pole_angle_spike_time[i] = firing_time(relative_angle, i);
        pole_velocity_spike_time[i] = firing_time(relative_angular_velocity, i);
        cart_position_spike_time[i] = firing_time(relative_cart, i);
        cart_velocity_spike_time[i] = firing_time(relative_cart_velocity, i);
//        io_printf(IO_BUF, "bin (p, pv, c, cv), %d (%k, %k, %k, %k)\n", i, (accum)pole_angle_spike_time[i],
//        (accum)pole_velocity_spike_time[i], (accum)cart_position_spike_time[i], (accum)cart_velocity_spike_time[i]);
    }

//    io_printf(IO_BUF, "motor force = %k\n", (accum)motor_force);
//    io_printf(IO_BUF, "max_pole_angle_bin = %k, abs = %k\n", (accum)max_pole_angle_bin, (accum)(abs(pole_angle)));
//    io_printf(IO_BUF, "pole (d,v,a):(%k, %k, %k) and cart (d,v,a):(%k, %k, %k)\n", (accum)pole_angle, (accum)pole_velocity,
//                        (accum)pole_acceleration, (accum)cart_position, (accum)cart_velocity, (accum)cart_acceleration);


    if (cart_position > track_length || cart_position < 0  || pole_angle > max_pole_angle  || pole_angle < min_pole_angle) {
        io_printf(IO_BUF, "failed out\n");
        return false;
    }
    else{
        read_force = true;
        return true;
    }
}

void mc_packet_received_callback(uint keyx, uint payload)
{
    // make this bin related for rank encoding, relate to force increments
    uint32_t compare;
    compare = keyx & 0x20;
//    io_printf(IO_BUF, "compare = %x\n", compare);
    use(payload);
    if (read_force){
        motor_force = ((float)compare * force_increment) + min_motor_force;
        read_force = false;
    }
//    if(compare == BACKWARD_MOTOR){
//        motor_force = motor_force - force_increment;
//        if (motor_force < min_motor_force){
//            motor_force = min_motor_force;
//        }
//    }
//    else if(compare == FORWARD_MOTOR){
//        motor_force = motor_force + force_increment;
//        if (motor_force > max_motor_force){
//            motor_force = max_motor_force;
//        }
//    }
}

float rand021(){
    return (float)(mars_kiss64_seed(kiss_seed) / (float)0xffffffff);
}

float norm_dist(float mean, float stdev){
    accum norm_dist;
    norm_dist = gaussian_dist_variate(mars_kiss64_simp, NULL);
    norm_dist = (norm_dist * stdev) + mean;
    return (float)norm_dist;
}

bool firing_prob(float relative_value, int bin){
    float norm_value = norm_dist(0, bin_width / bin_overlap);
    float separation = relative_value - (bin_width * (float)bin);
    if (separation < 0){
        separation = -separation;
    }
//    io_printf(IO_BUF, "norm = %k, separation = %k, realtive = %k, bin = %d\n", (accum)norm_value, (accum)separation, (accum)relative_value, bin);
    if (norm_value < 0){
        norm_value = -norm_value;
    }
    if (norm_value > separation){
        if (rand021() < max_firing_prob){
            return true;
        }
        else{
            return false;
        }
    }
    else{
        return false;
    }
}

void send_status(){
    for (int i = 0; i < number_of_bins; i = i + 1){
        pole_angle_spike_time[i] = pole_angle_spike_time[i] - 1.f;
        if (pole_angle_spike_time[i] <= 0){
            pole_angle_spike_time[i] = 1000.f;
            spike_angle(i);
        }
        pole_velocity_spike_time[i] = pole_velocity_spike_time[i] - 1.f;
        if (pole_velocity_spike_time[i] <= 0){
            pole_velocity_spike_time[i] = 1000.f;
            spike_angle_v(i);
        }
        cart_position_spike_time[i] = cart_position_spike_time[i] - 1.f;
        if (cart_position_spike_time[i] <= 0){
            cart_position_spike_time[i] = 1000.f;
            spike_cart(i);
        }
        cart_velocity_spike_time[i] = cart_velocity_spike_time[i] - 1.f;
        if (cart_velocity_spike_time[i] <= 0){
            cart_velocity_spike_time[i] = 1000.f;
            spike_cart_v(i);
        }
    }
}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks)
    {
        //spin1_pause();
        recording_finalise();
        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);
        //    spin1_callback_off(MC_PACKET_RECEIVED);

        io_printf(IO_BUF, "infinite_run %d; time %d\n",infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);
        //    io_printf(IO_BUF, "key count Left %u\n", left_key_count);
        //    io_printf(IO_BUF, "key count Right %u\n", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
//        simulation_handle_pause_resume(NULL);
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        if (_time == 0){
            update_state(0);
            // possibly use this to allow updating of time whenever
//            auto start = chrono::steady_clock::now();
        }
        // Increment ticks in frame counter and if this has reached frame delay
        tick_in_frame++;
        if(tick_in_frame == time_increment)
        {
            if (in_bounds){
                max_balance_time = (float)_time;
//                max_balance_time = max_balance_time + 1;
                in_bounds = update_state((float)time_increment / 1000.f);
            }
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 0.1s
            if(score_change_count >= 100){
                current_state[0] = cart_position;
                current_state[1] = pole_angle;
                if(reward_based == 0){
                    recording_record(0, &current_state, 8);
                }
                else{
                    recording_record(0, &max_balance_time, 4);
                }
                score_change_count=0;
            }
        }
        if (in_bounds){
            send_status();
        }
    }
//    io_printf(IO_BUF, "time %u\n", ticks);
//    io_printf(IO_BUF, "time %u\n", _time);
}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


//-------------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    io_printf(IO_BUF,"Error in initialisation - exiting!\n");
    rt_error(RTE_SWERR);
    return;
  }

  tick_in_frame = 0;

  // Set timer tick (in microseconds)
  io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
  spin1_set_timer_tick(timer_period);

  io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);
  io_printf(IO_BUF, "timer tick %d, %k\n",TIMER_TICK, (accum)TIMER_TICK);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  _time = UINT32_MAX;

  simulation_run();




}
