<!DOCTYPE html>
<html>
<head>
	<title>Newsletter</title>
	<link rel="stylesheet" type="text/css" href="style.css">
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7" crossorigin="anonymous">
	<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js" integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS" crossorigin="anonymous"></script>	
</head>
<body>
	<div class="container">
		<h1>Druva Machine Learning Project</h1>
		<hr>
		<a href="/model" style=" margin-right:20px;">Model Analysis</a>
		
		<a href="/optimizer" style=" margin:20px">Optimizers Analysis</a>

		<hr>
				
		<form class="form-vertical" action="/output" method="POST" enctype="multipart/form-data">
			<div class="row">	
				<div class="form-group col-md-4">	
					<label class="control-label">Window/Sequence Length:</label>
					<input class="form-control" type="number" name="wl">
				</div>

				<div class="form-group col-md-4">
					<label class="control-label">Future Predictions:</label>
					<input class="form-control" type="number" name="fp">
				</div>

				<div class="form-group col-md-4">
					<label class="control-label">Choose CSV file:</label>
					<input type="file" name="csvip">
				</div>
			</div>

			<div class="form-group">
				<label class="control-label">CSV file:</label>
				<input class="form-control" type="text" name="csvip2">
			</div>	
				
			<br>
			<input class="btn btn-primary" type="submit">
		</form>

		<br>
		<br> 
		<hr> </hr>
		<img  src="/home/abhi/Druva/final_22june/finaltraincode/{{picture}}" class="img-responsive" alt="Mountain View">
	</div>
</body>
</html
