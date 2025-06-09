# PowerShell script to test the GA Solver Chatbot API endpoint.
# This script simulates a user conversation to test session state.

# --- Configuration ---
$ApiUrl = "http://127.0.0.1:8000/chat"
$UserId = "powershell_tester"
# Generate a unique session ID for each run of the script to ensure a fresh conversation
$SessionId = "session_" + (Get-Random -Maximum 99999)

# --- Helper Function to Make API Calls ---
function Invoke-ChatApi {
    param(
        [string]$Message,
        [string]$CurrentSessionId
    )

    $body = @{
        user_id = $UserId
        session_id = $CurrentSessionId
        message = $Message
    } | ConvertTo-Json

    Write-Host "--- Sending Request ---"
    Write-Host "Message: $Message"
    # Write-Host "Body: $body" # Uncomment for verbose debugging

    try {
        # Using Invoke-RestMethod automatically parses the JSON response into a PowerShell object
        $response = Invoke-RestMethod -Uri $ApiUrl -Method Post -ContentType "application/json" -Body $body
        
        Write-Host "--- Received Response ---"
        # Format-List provides a nice, readable output for the response object
        $response | Format-List
        Write-Host "------------------------`n"
    }
    catch {
        Write-Host "!!! API Call Failed !!!" -ForegroundColor Red
        Write-Host "Error making POST request to $ApiUrl"
        Write-Host $_.Exception.Message
        Write-Host "Please ensure the FastAPI server is running (`uvicorn main:app --reload`)"
        Write-Host "------------------------`n"
        # Exit the script if the API is not running to prevent further errors
        exit 1
    }
}

# --- Test Sequence ---
Write-Host "Starting API test sequence for session: $SessionId" -ForegroundColor Green
Write-Host "=================================================="
Start-Sleep -Seconds 1

# Step 1: Set the population size for the session.
Write-Host "Step 1: Setting population size to 100..." -ForegroundColor Yellow
Invoke-ChatApi -Message "set population to 100" -CurrentSessionId $SessionId
Start-Sleep -Seconds 2

# Step 2: Ask to solve the TSP. The backend should use the population size of 100 set in Step 1.
Write-Host "Step 2: Asking solver to run (should use population of 100)..." -ForegroundColor Yellow
Invoke-ChatApi -Message "solve tsp" -CurrentSessionId $SessionId

Write-Host "=================================================="
Write-Host "API test sequence completed." -ForegroundColor Green

# To run this script:
# 1. Open a PowerShell terminal.
# 2. Navigate to the 'ga-solver-simple/backend' directory.
# 3. If you see an error about script execution being disabled, you may need to run:
#    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# 4. Ensure the FastAPI server is running in another terminal.
# 5. Execute the script: .\test_api.ps1 