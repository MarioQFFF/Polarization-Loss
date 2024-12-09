import numpy as np
import math

def normalize(v):
    return v / np.dot(v, v)**.5

def project_away(v, N):
    #project away N from v, assumes N normalized
    return v - N * np.dot(v, N)

def reflect_through(v, N):
    #reflect v through N, assumes N normalized
    #output preserves the norm of v
    return v - 2 * N * np.dot(v, N)


def plane_angle(V1, V2, nudges_size, nudges_number):
    #Calculate the normal vector of the original plane of incidence.
    n_o = np.cross(V1,V2)/np.linalg.norm(np.cross(V1,V2))
    n_s = []
    n_s.append(n_o)
    for i in range(nudges_number):
        #Incorporate the nuge to V1
        V1_nudge =(V1 + (nudges_size*(i+1)))/(np.linalg.norm(V1 + nudges_size*(i+1)))
        #If necessary, incorporate the nudge on V2
        #V2_nudge =(V2 -2* (nudges_size*(i+1)))/(np.linalg.norm(V2 - 2*nudges_size*(i+1)))
        #Calculate the normal vector of the new plane of incidence.
        n_i = np.cross(V1_nudge, V2)/np.linalg.norm(np.cross(V1_nudge, V2))
        n_s.append(n_i)
    plane_angles = []
    for n in n_s:
        #calculate the angles
        angle = np.arccos(np.dot(n_o, n))
        plane_angles.append(angle)
    return plane_angles


def polarization_circulation(pos_xyx_list, J_i, Rp, Rs, WaveLength, e1):
    p_s = np.array(pos_xyx_list)
    
    #generate beam direction vectors from list of positions
    PointinV = np.roll(p_s, -1, axis=0) - p_s
    
    v_s = np.array([v / np.linalg.norm(v) for v in PointinV[:-1]])
    
    print('Direction vectors:\n' +str(v_s))
    
    #Create v_prev to define N
    v_next = np.roll(v_s, -1, axis=0)
    
    #Generate mirror normals from beam direction vectors
    normals = v_next[:-1] - v_s[:-1]
    #Store the normal vectors N
    N_s = [-n/np.linalg.norm(n) for n in normals]
    
    
    #The Jones Matrices of the mirrors will be stored here
    reflectCoef = [] 
    #Check if the wavelength is none
    if WaveLength != None:  
        WaveLength2 = WaveLength/1000000000
    else: WaveLength2 =1
    
    #Now we generate the Jones matrices of the mirrors
    for v in range(len(v_s[:-1])):
        #Calculate the Cosine of beta
        cos_beta = (1 - np.dot(v_next[v],v_s[v]))/np.dot((v_next[v] - v_s[v]), (v_next[v] - v_s[v]))**.5
        
        #Calculate the sine of beta
        sin_beta = np.linalg.norm(np.cross(v_s[v], v_next[v])/np.linalg.norm(v_next[v]-v_s[v]))
        
        #Then we find the angle of incidence beta
        beta = np.arccos(cos_beta)
        print(f"Angle of incidence {v}: {beta}")
        #Check if user  input reflection coefficients
        if Rs is None:
            #Permitivity of the mirror as a fraction of the permitivity of free space
            ef= e1
            #Wave number
            k0 = (math.pi*2)/WaveLength2
        
            #Define incident wave vector k_i:
            k_i = k0*v_s[v]
        
            #Define reflected wave vector k_r:
            k_r = k0*v_next[v]
        
            #Define vectors a and b to incorporate Fresnel´s equations in coordinate free form (Can be found on the derivation in appendix I, Derivation of Fresnel´s equations in coordinate free form.)
            a = np.cross(k_i,(N_s[v]))
            b = np.cross((N_s[v]),a)
            #Define the transmitted wave vector
            k_t = (b + math.sqrt((k0**2)*ef-(k0**2)*sin_beta**2)*(N_s[v]))/np.linalg.norm((b + math.sqrt((k0**2)*ef-(k0**2)*sin_beta**2)*(N_s[v])))
            
            #Define the refection coefficients for S and P polarizations
            rs = ((np.dot(a, np.cross(k_t,k_i)))/(np.dot(a,np.cross(k_r,k_t))))
            rp = (np.dot(normalize(k_r),normalize(k_t))/np.dot(normalize(k_i),normalize(k_t)))*rs
            print('Wave number:', k0)
        else:
            rs = Rs
            rp = Rp
        # Create the Jone Matrix of the mirror.
        A = np.array([[rp, 0], [0, rs]])
        reflectCoef.append(A) #Store each Jones Matrix
        
    print('Normal Vectors:\n'+str(N_s))
    
    #Store P and S incident and reflected polarizations.
    P_pols = []
    P_pol_ref = []
    S_pols = []
    S_pol_ref = []
    for i, (v, v_n) in enumerate(zip(v_s[:-1], v_next[:-1])):
        #Generate canonical p polarization axis for each mirror (Must lay on the incidence plane defined by V_i and N and must be orthogonal to V_i).
        pol_p = np.where(np.isclose(np.abs(np.dot(v, v_n)), 1.0),
                            P_pol_ref[i-1] if i > 0 else np.zeros_like(v),  # Use previous reflected polarization or zero for first step
                            project_away(v_n, v) / np.linalg.norm(project_away(v_n, v)) if np.linalg.norm(project_away(v_n, v)) > 0 else P_pol_ref[i-1] ) 
        #Append the normalized polarization axis P to the list of polarizations.
        P_pols.append(pol_p)
        #Now we find the reflected polarization axis. 
        Pr = np.where(np.isclose(np.abs(np.dot(v, v_n)), 1.0),
                            P_pols[i] if i > 0 else np.zeros_like(v),  # Use previous incident polarization or zero for first step
                            project_away(v, v_n) / np.linalg.norm(project_away(v, v_n)) if np.linalg.norm(project_away(v, v_n)) > 0 else P_pols[i-1])
        #Incorporate into the list of reflected polarizations.
        P_pol_ref.append(Pr)
        #calculate s polarization
        pol_s = np.cross(pol_p, v)
        S_pols.append(pol_s)   
        #calculate reflected s polarization
        Sr = np.cross(Pr, v_n)
        S_pol_ref.append(Sr)
        #Print P_pol and P_pol_refl for debugging if needed
        print(f"Iteration {i}: P_pol = {pol_p}, P_pol_refl = {Pr}")
    #Now print all P and S polarization vectors as well as the reflected.
    print('P polarization Vectors:\n'+str(P_pols))
    print('S polarization Vectors:\n'+str(S_pols))
    print('P reflected polarization Vectors:\n'+str(P_pol_ref))
    print('S reflected polarization Vectors:\n'+str(S_pol_ref)) 
    #Check that the P and S polarizations are orthogonal to V_s 
    PdotV = np.einsum('ij,ij->i', P_pols, v_s[:-1])
    print("Check that P-pol vectors are orthogonal to v_s:\n"+str(PdotV))
    SdotV = np.einsum('ij,ij->i', S_pols, v_s[:-1])
    print("Check that S-pol vectors are orthogonal to v_s:\n"+str(SdotV))
    
    #Store the rotation matrices R.
    Rotation_matrices = []
    
    for i in range(len(P_pol_ref)-1):
        j = i+1
        #Using the reflected polarization and the polarization of the next wave  we find each element of the rotation matrix.
        PP = np.dot(P_pols[j], P_pol_ref[i])
        PS = np.dot(P_pols[j], S_pol_ref[i])
        SP = np.dot(S_pols[j], P_pol_ref[i])
        SS = np.dot(S_pols[j], S_pol_ref[i])
        
        #Create the rotation matrix
        R = np.array([
            [PP, PS],
            [SP, SS]
            ])
        #Add the rotation matrix to the list.
        Rotation_matrices.append(R)
    print('Jones matrices of the mirrors J:\n'+str(reflectCoef))
    print('Rotation Matrices R:\n'+str(Rotation_matrices))
    
    #Genreate the final polarization vector.
    J_f = J_i
    Rotation_matrices.append(np.eye(2))
    #Calculate the final polarization by acting each reflection and rotation matrix in order on the initial polarization.
    for i, (rf, rm) in enumerate(zip(reflectCoef, Rotation_matrices)):  
        #Multiply J_f by the reflection matrix A first and then the rotation matrix R, then normalize.
        J_f = np.dot(rm, np.dot(rf, J_f))
        print(J_f)
    J_iT = J_i.reshape(1, -1)
    J_fT = J_f.reshape(1, -1)/np.linalg.norm(J_f)
    
    epsilon_pol = (1 - np.abs(np.dot(J_iT, J_f)))
    print('Polarization loss:\n' + str(epsilon_pol))
    angles =total_polarization_angle(J_iT, J_fT)
    #Return the initial and final polarizations.
    return  J_i, J_f, angles

def total_polarization_angle(E1, E2):
    #Calculate the angle between the initial and final polarizations.
    psi = np.arccos((np.dot(E1, E2.T)))
    return psi

if __name__ == '__main__':
    #square of mirrors alternating
    #Now we create the Jones vector of the beam of light, with a linear polarization state
    V1 = np.array([0, -2, 0])
    V2 = np.array([1, 0, 0])
    nudge = np.array([0, 0 , 0.0125])
    angles = plane_angle(V1, V2, nudge, 28)
    print('Plane angles')
    print(angles)
    k = 28
    E_i = np.array([[1, 0]]).T
    n = 0.0125
    
    positions = [ (-1, 3, 1 ), (-1, 1, 1 ), (0, 1, 1), (0, 1, 0 ), (0, 0, 0),] 
    nudges = ([0,0,0],[0,0,0.0125*x],[0,0,0.0125*x],[0,0,0],[0,0,0])
    new_positions= list(map(np.add,positions,nudges))
    p_lambda = 1064
    permittivity = 3.9
    jones = polarization_circulation(positions, E_i, 1, 1, p_lambda, permittivity)
    J1 = jones[0].T
    J2 = jones[1].T
    angless = jones[-1]
    print('Initial Polarization:')
    print(J1)
    print('Final Polarization:')
    print(J2)
    print('Total Polarization angle: ')
    print(angless)
