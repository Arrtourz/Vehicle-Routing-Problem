# VehicleRoutingProblem
    ''' Class containing the data samples organized as follows:
        {p_i, v_i, y_i}_{i \in [N]}

        where p_i are the samples associated to passengers:

        p_i = <pOL^j, pOT^j, pDL^j, pDT^j>_{j \in R}
        where
            - pOL^j = Origin Latitude of request j
            - pOT^j = Origin Longitude of request j
            - pDL^j = Destination Latitude of request j
            - pDT^j = Destination Longitude of request j
            R = set of requests

        v_i = <vOL^j, vOT^j>_{j \in V}
        where
            - vOL^j = Origin Latitude of vehicle j
            - vOT^j = Origin Longitude of vehicle j
            V = set of vehicles

        y_i \in {0,1}^{|R| x |V|}
            is a 0-1 matrix whose entry:

            y_i[j,k] = 1 <->  request j is associated to vehicle k
    '''
