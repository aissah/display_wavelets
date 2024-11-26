"""Supporting Functions for coherence analysis of DAS data."""
# import h5py
import numpy as np

def windowed_spectra(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Calculate the frequency domain representation of data in windows.

    Parameters
    ----------
    data : numpy array
        Data in time domain
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to return the spectra at. The default is None.
        If None, the spectra is returned at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    win_spectra : numpy array
        Spectra of the data in windows
    frequencies : numpy array
        Frequencies at which the spectra is computed

    """
    win_start = 0
    window_samples = int(subwindow_len / sample_interval)
    total_samples = data.shape[-1]
    overlap = int(overlap / sample_interval)
    intervals = np.arange(
        window_samples, total_samples + 1, window_samples, dtype=int
    )  # break time series into windowed intervals

    win_end = intervals[0]

    spectra = np.fft.rfft(data[:, win_start:win_end])
    win_spectra = spectra[np.newaxis]

    while win_end < total_samples:
        win_start = win_end - overlap
        win_end = win_start + window_samples
        spectra = np.fft.rfft(data[:, win_start:win_end])
        win_spectra = np.append(win_spectra, spectra[np.newaxis], axis=0)
        # win_start = win_end

    frequencies = np.fft.rfftfreq(window_samples, sample_interval)

    return win_spectra, frequencies


def normalised_windowed_spectra(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Compute frequency domain representation of data nomralized in windows.

    Parameters
    ----------
    data : numpy array
        Data in time domain
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to return the spectra at. The default is None.
        If None, the spectra is returned at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    normalized_spectra : numpy array
        Normalized spectra of the data. The normalization is done
        by dividing the spectra by the sum of the absolute values of the
        spectra squared of each channel
    frequencies : numpy array
        Frequencies at which the spectra is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    mean_spectra = np.mean(win_spectra, axis=0)
    win_spectra -= mean_spectra

    normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
    normalizer = np.tile(np.sqrt(normalizer), (win_spectra.shape[0], 1, 1))
    normalizer = normalizer.transpose(2, 1, 0)

    normalized_spectra = win_spectra.transpose(2, 1, 0) / normalizer

    return normalized_spectra, frequencies


def welch_coherence(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Calculate the coherence matrix at all frequencies. 
    
    The welch method is used for spectra density calculation.

    Parameters
    ----------
    data : numpy array
        Data in time for coherence analysis
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to compute the coherence at. The default is
        None. If None, the coherence is computed at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    coherence : numpy array
        Coherence matrix of the data
    frequencies : numpy array
        Frequencies at which the coherence is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    mean_spectra = np.mean(win_spectra, axis=0)
    win_spectra -= mean_spectra

    normalizer = np.sum(np.absolute(win_spectra) ** 2, axis=0)
    normalizer = np.tile(normalizer, (normalizer.shape[0], 1, 1))
    normalizer = normalizer * normalizer.transpose((1, 0, 2))
    normalizer = normalizer.transpose(2, 1, 0)

    welch_numerator = np.matmul(
        win_spectra.transpose(2, 1, 0),
        np.conjugate(win_spectra.transpose(2, 0, 1)),
    )
    welch_numerator = np.absolute(welch_numerator) ** 2
    coherence = np.multiply(welch_numerator, 1 / normalizer)

    return coherence, frequencies


def covariance(
    data: np.array,
    subwindow_len: int,
    overlap: float,
    freq=None,
    sample_interval: int = 1,
) -> tuple:
    """
    Calculate the covariance matrix at all frequencies.

    Parameters
    ----------
    data : numpy array
        Data in time for covariance analysis
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    freq : int, optional
        Frequency to compute the covariance at. The default is
        None. If None, the covariance is computed at all frequencies
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    covariance : numpy array
        Covariance matrix of the data
    frequencies : numpy array
        Frequencies at which the coherence is computed

    """
    win_spectra, frequencies = windowed_spectra(
        data, subwindow_len, overlap, freq, sample_interval
    )

    covariance = np.matmul(
        win_spectra.transpose(2, 1, 0),
        np.conjugate(win_spectra.transpose(2, 0, 1)),
    )
    # welch_numerator = np.absolute(welch_numerator) ** 2

    return covariance, frequencies


def exact_coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int = 0,
    resolution: float = 0.1,
    sample_interval: int = 1,
):
    """
    Compute the detection significance from coherence.
    
    The detection significance is the ratio of the largest eigenvalue
    to the sum of all eigenvalues. This method computes the coherence matrix
    using the Welch method, and then computes the eigenvalues and subsequent
    detection significance at all frequencies.

    Parameters
    ----------
    data : numpy array
        Data in time for coherence analysis.
    subwindow_len : int
        Length of the subwindows in seconds.
    overlap : int, optional
        Overlap between adjacent subwindows in seconds.
        The default is 0.
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 0.1.
    sample_interval : float, optional
        Sample interval of the data. The default is 1.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the exact method
    eigenvalss : numpy array
        Eigenvalues of the coherence matrix

    """
    coherence, _ = welch_coherence(
        data, subwindow_len, overlap, sample_interval=sample_interval
    )
    num_frames = coherence.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    num_subwindows = coherence.shape[2]
    detection_significance = np.empty(num_frames)
    # store the eigenvalues
    eigenvalss = np.empty((num_frames, num_subwindows))
    freq_interval = int(1 / resolution)

    for d in range(num_frames):
        # eigenvals, _ = np.linalg.eig(coherence[d * freq_interval])
        eigenvals = np.linalg.eigvalsh(coherence[d * freq_interval])
        eigenvalss[d] = eigenvals[:num_subwindows]
        eigenvals = np.sort(eigenvals)[::-1]
        detection_significance[d] = eigenvals[0] / np.sum(eigenvals)

    return detection_significance, eigenvalss


def svd_coherence(norm_win_spectra: np.ndarray, resolution: float = 1):
    """
    Compute the detection significance from SVD approximation of coherence.
    
    The detection significance is the ratio of the largest
    eigenvalue to the sum of all eigenvalues. This method computes the
    coherence matrix from the normalised spectra matrix provided, and then
    approximates the eigenvalues and subsequent detection significance at
    all frequencies using SVD.

    Parameters
    ----------
    norm_win_spectra : numpy array
        Normalized windowed spectra
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 0.1.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the SVD method
    svd_approxs : numpy array
        Approximation of the eigenvalues of the data using the
        SVD method

    """
    num_frames = norm_win_spectra.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    num_subwindows = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    svd_approxs = np.empty((num_frames, num_subwindows))
    freq_interval = int(1 / resolution)

    for d in range(num_frames):
        singular_values = np.linalg.svd(
            norm_win_spectra[d * freq_interval],
            compute_uv=False,
            hermitian=False,
        )
        svd_approx = singular_values**2
        svd_approxs[d] = svd_approx[:num_subwindows]
        detection_significance[d] = svd_approx[0] / np.sum(svd_approx)

    return detection_significance, svd_approxs


def qr_coherence(norm_win_spectra: np.ndarray, resolution: float = 1):
    """
    Compute the detection significance from QR decomposition approximation of coherence.
    
    The detection significance is the ratio of the
    largest eigenvalue to the sum of all eigenvalues. This method computes the
    coherence matrix from the normalised spectra matrix provided, and then
    approximates the eigenvalues and subsequent detection significance at all
    frequencies using QR decomposition.

    Parameters
    ----------
    norm_win_spectra : numpy array
        Normalized windowed spectra
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 0.1.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using
        the QR decomposition
    qr_approxs : numpy array
        Approximation of the eigenvalues of the data using the QR
        decomposition

    """
    num_frames = norm_win_spectra.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    # num_subwindows = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    qr_approxs = np.empty((num_frames, np.min(norm_win_spectra.shape[1:])))
    freq_interval = int(1 / resolution)

    for d in range(num_frames):
        _, r_matrix = np.linalg.qr(norm_win_spectra[d * freq_interval])
        qr_approx = np.diag(r_matrix @ np.conjugate(r_matrix.transpose()))
        sorted_qr_approx = np.sort(qr_approx)[::-1]
        detection_significance[d] = sorted_qr_approx[0] / np.sum(
            np.absolute(sorted_qr_approx)
        )
        qr_approxs[d] = qr_approx

    return detection_significance, qr_approxs


def rsvd_coherence(
    norm_win_spectra: np.ndarray, resolution: int = 1, approx_rank: int = None
):
    """
    Compute the detection significance from randomized SVD approximation of coherence.
    
    The detection significance is the ratio
    of the largest eigenvalue to the sum of all eigenvalues. This method
    computes the coherence matrix from the normalised spectra matrix provided,
    and then approximates the eigenvalues and subsequent detection significance
    at all frequencies using randomized SVD.

    Parameters
    ----------
    norm_win_spectra : numpy array
        Normalized windowed spectra
    resolution : float, optional
        Resolution of the detection significance from 0 to 1.
        The default is 0.1.
    approx_rank : int, optional
        Approximate rank for the randomized SVD method.
        The default is None.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the data based on coherence
        computed using the randomized SVD method
    rsvd_approxs : numpy array
        Approximation of the eigenvalues of the data using the
        randomized SVD method

    """
    from sklearn.utils.extmath import randomized_svd  # type: ignore

    num_frames = norm_win_spectra.shape[0]
    num_frames = int(num_frames * resolution)

    # Custom line due to apparent lowpass in BH data:
    # only use 3/5 of the frames
    num_frames = int(num_frames * 2 / 5)

    if approx_rank is None:
        approx_rank = norm_win_spectra.shape[2]
    detection_significance = np.empty(num_frames)
    rsvd_approxs = np.empty((num_frames, approx_rank))

    for d in range(num_frames):
        _, singular_values, _ = randomized_svd(norm_win_spectra[d], approx_rank)
        rsvd_approx = singular_values**2
        rsvd_approxs[d] = rsvd_approx
        detection_significance[d] = rsvd_approx[0] / np.sum(rsvd_approx)

    return detection_significance, rsvd_approxs


def qr_iteration(matrix: np.array, tol: float = 1e-6, max_iter: int = 1000) -> np.array:
    """
    Compute the eigenvalues of matrix using the QR iteration method.

    Parameters
    ----------
    matrix : numpy array
        Matrix to compute the eigenvalues of
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    numpy array
        Eigenvalues of matrix

    """
    n = matrix.shape[0]
    q_matrix = np.eye(n)
    for i in range(max_iter):
        q_matrix, r_matrix = np.linalg.qr(matrix)
        matrix = r_matrix @ q_matrix
        if np.linalg.norm(np.tril(matrix, -1)) < tol:
            break
    return np.diag(matrix)


def power_iteration(matrix: np.array, tol: float = 1e-6, max_iter: int = 1000) -> float:
    """
    Compute first eigenvalue of matrix using the power iteration method.

    Parameters
    ----------
    matrix : numpy array
        Matrix to compute the eigenvalues of
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    float
        Largest eigenvalue of matrix

    """
    n = matrix.shape[0]
    x = np.random.rand(n)
    for i in range(max_iter):
        new_x = matrix @ x
        new_x = new_x / np.linalg.norm(new_x)
        if np.linalg.norm(new_x - x) < tol:
            x = new_x
            break
        x = new_x
    return x @ matrix @ x


def coherence(
    data: np.array,
    subwindow_len: int,
    overlap: int,
    resolution: float = 1,
    sample_interval: float = 1,
    method: str = "exact",
    approx_rank: int = 10,
):
    """
    Compute a detection significance using coherence.

    Parameters
    ----------
    data : numpy array
        Data for coherence analysis
    subwindow_len : int
        Length of the subwindows in seconds
    overlap : int
        Overlap between adjacent subwindows in seconds
    Freq : int, optional
        Frequency to compute the coherence at, option is not
        implemented yet. The default is None. If None, the coherence is
        computed at all frequencies.
    sample_interval : float, optional
        Sample interval of the data. The default is 1.
    method : str, optional
        Method to use for coherence analysis.
        The default is 'exact'.
        Options are: 'exact', 'qr', 'svd', 'rsvd', 'power', 'qr iteration'
    approx_rank : int, optional
        Approximate rank for the randomized SVD method.
        The default is 10.

    Returns
    -------
    detection_significance : numpy array
        Detection significance of the coherence of the data
        computed using the specified method

    Example
    --------
    data = np.random.rand(100, 1000)
    detection_significance = coherence(data, 10, 5, method='exact')
    """
    METHODS = ["exact", "qr", "svd", "rsvd", "power", "qr iteration"]

    if method == "exact":
        return exact_coherence(
            data,
            subwindow_len,
            overlap,
            sample_interval=sample_interval,
            resolution=resolution,
        )
    elif method == "qr":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return qr_coherence(norm_win_spectra, resolution=resolution)
    elif method == "svd":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return svd_coherence(norm_win_spectra, resolution=resolution)
    elif method == "rsvd":
        norm_win_spectra, _ = normalised_windowed_spectra(
            data, subwindow_len, overlap, sample_interval=sample_interval
        )
        return rsvd_coherence(
            norm_win_spectra, resolution=resolution, approx_rank=approx_rank
        )
    elif method == "power":
        return power_iteration(data, tol=1e-6, max_iter=1000)
    elif method == "qr iteration":
        return qr_iteration(data, tol=1e-6, max_iter=1000)
    else:
        error_msg = f"Invalid method: {method}; valid methods are: {METHODS}"
        raise ValueError(error_msg)


def rm_laser_drift(data: np.array) -> np.array:
    """
    RSemove laser drift from DAS data.
    
    We do this by subtracting the median of each time
    sample across the channels from each channel at that time
    sample. This assumes the first dimension of the data is
    along the fibre.

    Parameters
    ----------
    data : numpy array
        Data to remove laser drift from

    Returns
    -------
    data : numpy array
        Data with laser drift removed

    Example
    --------
    data = np.random.rand(100, 1000)
    data = rm_laser_drift(data)
    """
    # compute median along fibre for each time sample
    med = np.median(data, axis=0)
    # subtract median from each corresponding time sample
    data = data - med[np.newaxis, :]

    return data