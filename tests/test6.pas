 program ExampleProgram;
    var
        age, height: integer;
        name: string;
        salary: real;
        isActive: boolean;
    begin
        { Assigning values to variables }
        age := 25;
        height := 180;
        name := 'John Doe';
        salary := 2500.50;
        isActive := true;
        
        { Display values }
        writeln('Name: ', name);
        writeln('Age: ', age);
        writeln('Height: ', height, ' cm');
        writeln('Salary: $', salary:0:2);
        
        if isActive then
            writeln('Status: Active')
        else
            writeln('Status: Inactive');
    end.