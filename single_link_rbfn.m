clear; clc;

% Parameters
m = 0.02; 
l = 0.05; 
g = 9.8;

% Functions for dynamics
M = @(q) 0.1 + 0.06*sin(q);
C = @(q, q_dot) 3*q_dot + 3*cos(q);
G = @(q) m*g*l*cos(q);

% Initial conditions
q(1) = 0.15; 
dq(1) = 0;

% Desired trajectory functions
q_d = @(t) 10*sin(t);
dq_d = @(t) 10*cos(t);
ddq_d = @(t) -10*sin(t);

% Centers for radial basis functions
c_M = [-1.5 -1.0 -0.5 0 0.5 1.0 1.5];
c_C = [-1.5 -1.0 -0.5 0 0.5 1.0 1.5; -1.5 -1.0 -0.5 0 0.5 1.0 1.5];
c_G = [-1.5 -1.0 -0.5 0 0.5 1.0 1.5];

% Radial basis function width
b = 20;

% Controller gains
A = 0.8;  % Damping
Kp = 0.1;  % Proportional gain
Ki = 0.001;  % Integral gain
Kr = 0.0005;  % Control gain

% Initial weights for the neural network
W_M = zeros(1, 7);
W_C = zeros(1, 7);
W_G = zeros(1, 7);

% Learning rates for the neural network
Gamma_M = 0.005;  
Gamma_C = 0.005;  
Gamma_G = 0.005;  

% Time step and number of simulation steps
dt = 0.0001; 
num_steps = 300000;

% Preallocate storage for plotting
M_SNN_vals = zeros(1, num_steps);
C_SNN_vals = zeros(1, num_steps);
G_SNN_vals = zeros(1, num_steps);
tau_vals = zeros(1, num_steps);

M_actual_vals = zeros(1, num_steps);
C_actual_vals = zeros(1, num_steps);
G_actual_vals = zeros(1, num_steps);

% Main simulation loop
for i = 1:num_steps
    t = i * dt;  % Current time
    
    % Error between desired and actual positions
    e = q_d(t) - q(i);
    
    % Calculate velocity (dq) for current step
    if i > 1
        dq(i) = (q(i) - q(i-1)) / dt;
    end
    
    % Error in velocity
    de = dq_d(t) - dq(i);
    
    % Sliding mode variable
    r(i) = de + A * e;
    
    % Desired acceleration and velocity
    dq_r = dq_d(t) + A * e;
    ddq_r = ddq_d(t) + A * de;
    
    % State vector
    z = [q(i); dq(i)];
    
    % Calculate radial basis function outputs
    for j = 1:7
        H_M(j) = exp(-norm(q(i) - c_M(j)) / (b * b));
        H_C(j) = exp(-norm(z - c_C(:, j)) / (b * b));
        H_G(j) = exp(-norm(q(i) - c_G(j)) / (b * b));
    end
    
    % Neural network estimates
    M_SNN(i) = W_M * H_M';
    C_SNN(i) = W_C * H_C';
    G_SNN(i) = W_G * H_G';

    M_a(i) = 0.1 + 0.06*sin(q(i));
    C_a(i) = 3*dq(i) + 3*cos(q(i));
    G_a(i) = m*g*l*cos(q(i));
    
    % Ensure M_SNN is not zero
    if M_SNN(i) == 0
        M_SNN(i) = M_SNN(i) + 0.0001;
    end
    
    % Control input components
    tau_r = Kr * sign(r(i));
    tau_m = M_SNN(i) * ddq_r + C_SNN(i) * dq_r + G_SNN(i);
    
    % Integral of sliding mode variable
    integ = sum(r(1:i) * dt);
    
    % Total control input
    tau = tau_m + Kp * r(i) + Ki * integ + tau_r;
    
    % Update weights for neural network
    W_M = W_M + Gamma_M * H_M * ddq_r * r(i) * dt;
    W_C = W_C + Gamma_C * H_C * dq_r * r(i) * dt;
    W_G = W_G + Gamma_G * H_G * r(i) * dt;
    
    % Calculate acceleration
    ddq(i) = (tau - C_SNN(i) * dq(i) - G_SNN(i)) / M_SNN(i);
    
    % Update velocity and position
    dq(i+1) = dq(i) + ddq(i) * dt;
    q(i+1) = q(i) + dq(i+1) * dt;
    
    % Store values for plotting
    M_SNN_vals(i) = M_SNN(i);
    C_SNN_vals(i) = C_SNN(i);
    G_SNN_vals(i) = G_SNN(i);
    tau_vals(i) = tau;
    
    % Compute actual values for comparison
    M_actual_vals(i) = M(q(i));
    C_actual_vals(i) = C(q(i), dq(i));
    G_actual_vals(i) = G(q(i));
end

% Time vector for plotting
t = (0:num_steps) * dt;
q_d_vals = arrayfun(q_d, t);
dq_d_vals = arrayfun(dq_d, t);

% Plot the results
figure;
subplot(2,1,1);
plot(t, q, 'b', t, q_d_vals, 'r--');
legend('q', 'q_d');
xlabel('Time (s)');
ylabel('q');
title('q and q_d vs Time');

subplot(2,1,2);
plot(t(1:end-1), dq(1:end-1), 'b', t, dq_d_vals, 'r--');
legend('dq', 'dq_d');
xlabel('Time (s)');
ylabel('dq');
title('dq and dq_d vs Time');

figure;
subplot(3,1,1);
plot(t(1:end-1), M_SNN_vals, 'b', t(1:end-1), M_actual_vals, 'r--');
legend('M_{SNN}', 'M_{actual}');
xlabel('Time (s)');
ylabel('M');
title('Estimated and Actual M vs Time');

subplot(3,1,2);
plot(t(1:end-1), C_SNN_vals, 'b', t(1:end-1), C_actual_vals, 'r--');
legend('C_{SNN}', 'C_{actual}');
xlabel('Time (s)');
ylabel('C');
title('Estimated and Actual C vs Time');

subplot(3,1,3);
plot(t(1:end-1), G_SNN_vals, 'b', t(1:end-1), G_actual_vals, 'r--');
legend('G_{SNN}', 'G_{actual}');
xlabel('Time (s)');
ylabel('G');
title('Estimated and Actual G vs Time');

figure;
plot(t(1:end-1), tau_vals);
xlabel('Time (s)');
ylabel('\tau');
title('Control Input \tau vs Time');
